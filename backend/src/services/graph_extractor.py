"""
Legal Knowledge Graph Extractor

Builds a rich knowledge graph from already-parsed contract documents, inspired
by the Graphify approach: LLM-based extraction with provenance tracking and a
configurable legal-domain relationship schema.

Runs AFTER section_parser + entity_extractor.  Reads:
    fileSections, entityIndex, fileCrossRefs, documentRelationships, filePages

Writes (upsert-safe, re-runnable):
    graphNodes  — document / section / person / org nodes
    graphEdges  — typed, provenanced edges between nodes

Node types:
    document      one per filePages record
    section       one per fileSections record
    person        deduplicated across conversation (aliases collapsed)
    organization  deduplicated across conversation (aliases collapsed)

Edge types (legal domain):
    CONTAINS    Document → Section (structural)
    CHILD_OF    Section  → Section (parentSectionId hierarchy)
    REFERENCES  Section  → Section (explicit cross-references via fileCrossRefs)
    CONDITIONS  Section A is subject to / depends on Section B
    SUPERSEDES  Section/Doc A replaces / overrides Section/Doc B
    EXCEPTIONS  Section A carves out of or excludes from Section B
    DEFINES     Section  → defined term it introduces
    OBLIGATES   Section creates a binding obligation (shall/must)
    PERMITS     Section grants a right or permission (may/entitled)
    PROHIBITS   Section restricts an action (shall not/must not)
    PARTY_TO    Person/Org → Document (appears in that file)
    MASTER_OF   Master Document → Transaction Document
    NOVATES     Modification → Original Document
    TERMINATES  Termination Doc → Agreement
    RENEWS      Successor Doc → Predecessor Doc

Provenance tags (Graphify-style):
    EXTRACTED  — explicitly stated in document text
    INFERRED   — derived from context / structure
    AMBIGUOUS  — possible but uncertain

Public API:
    extract_graph(file_id, conversation_id)   → {nodes, edges, llm_edges}
    get_graph_for_conversation(conversation_id) → React-Flow-shaped dict
    get_section_graph(file_id, conversation_id) → React-Flow-shaped dict (drill-down)
    resolve_entity_aliases(conversation_id, query) → List[str]  (expanded entity names)
    get_graph_context_for_sections(conversation_id, seed_section_node_ids)
        → List[dict]  (sections linked via CONDITIONS / SUPERSEDES / EXCEPTIONS)
"""

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stable ID helpers
# ---------------------------------------------------------------------------

def _node_id(node_type: str, *parts: str) -> str:
    raw = f"{node_type}:" + ":".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _edge_id(from_id: str, edge_type: str, to_id: str) -> str:
    raw = f"{from_id}-{edge_type}-{to_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


# ---------------------------------------------------------------------------
# Entity deduplication  (Mainstay HK == Mainstay Asia Ltd)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = re.compile(
    r"\b(limited|ltd|llc|inc|corp|plc|pte|sdn|bhd|gmbh|ag|sa|nv|bv|oy|ab|"
    r"consulting|consultants|services|group|holdings|asia|pacific|international)\b",
    re.IGNORECASE,
)
_ARTICLES = re.compile(r"\b(the|a|an)\b", re.IGNORECASE)


def _entity_agg_key(name: str) -> str:
    """
    Aggressive normalization for entity deduplication.
    'Mainstay Asia Ltd' → 'mainstay'
    'Eames Consulting Limited' → 'eames'
    Keeps at least 2 tokens before stripping to avoid over-collapsing.
    """
    s = name.lower().strip()
    s = _ARTICLES.sub("", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    # Only strip suffixes if >1 meaningful token remains
    stripped = _LEGAL_SUFFIXES.sub("", s).strip()
    stripped_tokens = [t for t in stripped.split() if len(t) >= 3]
    return " ".join(stripped_tokens) if stripped_tokens else s


def _deduplicate_entities(conversation_id: str) -> Dict[str, dict]:
    """
    Read entityIndex, collapse aliases by aggressive key.
    Returns {agg_key: {nodeId, nodeType, label, aliases, fileIds}}
    """
    from src.utils.connection_utils import db

    records = list(db["entityIndex"].find(
        {"conversationId": conversation_id},
        {"entityType": 1, "entityValue": 1, "fileId": 1},
    ))

    groups: Dict[str, dict] = {}
    for rec in records:
        if rec.get("entityType") not in ("person", "organization"):
            continue
        val = (rec.get("entityValue") or "").strip()
        if not val:
            continue
        agg = _entity_agg_key(val)
        if not agg:
            continue
        nt = "person" if rec["entityType"] == "person" else "organization"
        if agg not in groups:
            groups[agg] = {
                "nodeType": nt,
                "aliases": set(),
                "fileIds": set(),
                "aggKey": agg,
            }
        groups[agg]["aliases"].add(val)
        groups[agg]["fileIds"].add(rec["fileId"])

    result: Dict[str, dict] = {}
    for agg, g in groups.items():
        canonical = max(g["aliases"], key=len)
        nid = _node_id(g["nodeType"], conversation_id, agg)
        result[agg] = {
            "nodeId": nid,
            "nodeType": g["nodeType"],
            "label": canonical,
            "aliases": sorted(g["aliases"]),
            "fileIds": sorted(g["fileIds"]),
        }
    return result


# ---------------------------------------------------------------------------
# LLM-based intra-clause relationship extraction
# ---------------------------------------------------------------------------

_RELATIONSHIP_PROMPT = """\
You are a legal knowledge graph expert. Analyse the contract sections below and \
extract semantic relationships between them.

Return ONLY a JSON array. Each element:
{{
  "from_section": "<sectionId>",
  "to_section":   "<sectionId>",
  "edge_type":    "<one of: CONDITIONS|SUPERSEDES|EXCEPTIONS|DEFINES|OBLIGATES|PERMITS|PROHIBITS>",
  "provenance":   "<EXTRACTED|INFERRED|AMBIGUOUS>",
  "confidence":   <float 0.0-1.0>,
  "detail":       "<one sentence>"
}}

Edge type definitions:
  CONDITIONS  — from_section only applies when conditions in to_section are met
                Keywords: "subject to", "provided that", "unless", "conditional on"
  SUPERSEDES  — from_section replaces / overrides to_section
                Keywords: "notwithstanding", "in lieu of", "replaces", "overrides", "prevails"
  EXCEPTIONS  — from_section carves out of to_section
                Keywords: "except as provided", "other than", "excluding", "save that"
  DEFINES     — from_section provides a definition used in to_section
  OBLIGATES   — from_section creates a binding obligation (shall/must) — to_section = same
  PERMITS     — from_section grants a right (may/entitled to) — to_section = same
  PROHIBITS   — from_section restricts an action (shall not / must not) — to_section = same

Rules:
  - Only output relationships with confidence >= 0.5
  - Both sectionIds must appear in the list below
  - Return [] if no clear relationships found

Document: {file_name}
Sections:
{sections_text}
"""


def _extract_clause_relationships(
    file_id: str,
    file_name: str,
    sections: List[dict],
) -> List[dict]:
    from src.utils.llm_utils import invoke_with_costing_evalution

    if not sections:
        return []

    lines = []
    for s in sections:
        preview = (s.get("content") or "")[:300].replace("\n", " ")
        lines.append(
            f"[{s['sectionId']}] ({s.get('clauseType','other')}) "
            f"{s.get('sectionTitle','')} — {preview}"
        )

    prompt = _RELATIONSHIP_PROMPT.format(
        file_name=file_name,
        sections_text="\n".join(lines),
    )

    try:
        response = invoke_with_costing_evalution(prompt=prompt)
        raw = (response.content or "").strip()
        if "```" in raw:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            raw = raw[start:end] if start != -1 and end > start else raw
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning("[GraphExtractor] LLM relationship extraction failed for %s: %s", file_name, e)
        return []


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def _build_document_node(file_id: str, conversation_id: str) -> Optional[dict]:
    from src.utils.connection_utils import db

    rec = db["filePages"].find_one(
        {"fileId": file_id},
        {"fileName": 1, "functionalRole": 1, "documentType": 1,
         "effectiveDate": 1, "documentRank": 1},
    )
    if not rec:
        return None
    return {
        "nodeId": _node_id("document", file_id),
        "conversationId": conversation_id,
        "nodeType": "document",
        "label": rec.get("fileName", file_id),
        "aliases": [],
        "fileId": file_id,
        "sectionId": None,
        "properties": {
            "functionalRole": rec.get("functionalRole", "standalone"),
            "documentType": rec.get("documentType", ""),
            "effectiveDate": rec.get("effectiveDate"),
            "documentRank": rec.get("documentRank", 0),
        },
    }


def _build_section_nodes(
    file_id: str, conversation_id: str, sections: List[dict]
) -> List[dict]:
    nodes = []
    for s in sections:
        nodes.append({
            "nodeId": _node_id("section", file_id, s["sectionId"]),
            "conversationId": conversation_id,
            "nodeType": "section",
            "label": s.get("sectionTitle") or s["sectionId"],
            "aliases": [],
            "fileId": file_id,
            "sectionId": s["sectionId"],
            "properties": {
                "clauseType": s.get("clauseType", "other"),
                "riskLevel": s.get("riskLevel", 1),
                "pageNumber": s.get("pageNumber"),
                "parentSectionId": s.get("parentSectionId"),
            },
        })
    return nodes


def _build_entity_nodes(entity_map: Dict[str, dict], conversation_id: str) -> List[dict]:
    nodes = []
    for agg_key, g in entity_map.items():
        nodes.append({
            "nodeId": g["nodeId"],
            "conversationId": conversation_id,
            "nodeType": g["nodeType"],
            "label": g["label"],
            "aliases": g["aliases"],
            "fileId": None,
            "sectionId": None,
            "properties": {"aggKey": agg_key, "fileIds": g["fileIds"]},
        })
    return nodes


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------

def _build_structural_edges(
    file_id: str, conversation_id: str, sections: List[dict]
) -> List[dict]:
    """CONTAINS + CHILD_OF from document/section hierarchy."""
    doc_nid = _node_id("document", file_id)
    edges = []
    for s in sections:
        sec_nid = _node_id("section", file_id, s["sectionId"])
        edges.append({
            "edgeId": _edge_id(doc_nid, "CONTAINS", sec_nid),
            "conversationId": conversation_id,
            "fromNodeId": doc_nid,
            "toNodeId": sec_nid,
            "edgeType": "CONTAINS",
            "provenance": "EXTRACTED",
            "confidence": 1.0,
            "sourceFileId": file_id,
            "sourceSectionId": None,
            "detail": f"Document contains section {s['sectionId']}",
        })
        parent = s.get("parentSectionId")
        if parent:
            parent_nid = _node_id("section", file_id, parent)
            edges.append({
                "edgeId": _edge_id(sec_nid, "CHILD_OF", parent_nid),
                "conversationId": conversation_id,
                "fromNodeId": sec_nid,
                "toNodeId": parent_nid,
                "edgeType": "CHILD_OF",
                "provenance": "EXTRACTED",
                "confidence": 1.0,
                "sourceFileId": file_id,
                "sourceSectionId": s["sectionId"],
                "detail": f"Section {s['sectionId']} is a sub-clause of {parent}",
            })
    return edges


def _build_crossref_edges(
    file_id: str, conversation_id: str, known_sec_ids: Set[str]
) -> List[dict]:
    """REFERENCES edges from resolved fileCrossRefs."""
    from src.utils.connection_utils import db

    edges = []
    seen: Set[str] = set()
    for ref in db["fileCrossRefs"].find({
        "sourceFileId": file_id,
        "conversationId": conversation_id,
        "resolvedFileId": {"$ne": None},
        "resolvedSectionId": {"$ne": None},
    }):
        src_sec = ref.get("sourceSectionId")
        tgt_file = ref.get("resolvedFileId")
        tgt_sec = ref.get("resolvedSectionId")
        if not (src_sec and tgt_file and tgt_sec):
            continue
        from_id = _node_id("section", file_id, src_sec)
        to_id = _node_id("section", tgt_file, tgt_sec)
        eid = _edge_id(from_id, "REFERENCES", to_id)
        if eid in seen:
            continue
        seen.add(eid)
        edges.append({
            "edgeId": eid,
            "conversationId": conversation_id,
            "fromNodeId": from_id,
            "toNodeId": to_id,
            "edgeType": "REFERENCES",
            "provenance": "EXTRACTED",
            "confidence": 0.9,
            "sourceFileId": file_id,
            "sourceSectionId": src_sec,
            "detail": f"Section {src_sec} references {ref.get('refClause')}",
        })
    return edges


def _build_doc_relationship_edges(conversation_id: str) -> List[dict]:
    """MASTER_OF / NOVATES / TERMINATES / RENEWS from documentRelationships."""
    from src.utils.connection_utils import db

    _MAP = {
        "master_child": "MASTER_OF",
        "novation": "NOVATES",
        "termination": "TERMINATES",
        "renewal": "RENEWS",
    }
    edges = []
    for rel in db["documentRelationships"].find({"conversationId": conversation_id}):
        from_file = rel.get("fromFileId")
        to_file = rel.get("toFileId")
        etype = _MAP.get(rel.get("relationshipType", ""))
        if not (from_file and to_file and etype):
            continue
        from_id = _node_id("document", from_file)
        to_id = _node_id("document", to_file)
        eid = _edge_id(from_id, etype, to_id)
        edges.append({
            "edgeId": eid,
            "conversationId": conversation_id,
            "fromNodeId": from_id,
            "toNodeId": to_id,
            "edgeType": etype,
            "provenance": "EXTRACTED",
            "confidence": 1.0,
            "sourceFileId": from_file,
            "sourceSectionId": None,
            "detail": rel.get("detail", ""),
        })
    return edges


def _build_entity_doc_edges(
    entity_map: Dict[str, dict], conversation_id: str
) -> List[dict]:
    """PARTY_TO: Person/Org → Document."""
    edges = []
    seen: Set[str] = set()
    for agg_key, g in entity_map.items():
        ent_nid = g["nodeId"]
        for fid in g["fileIds"]:
            doc_nid = _node_id("document", fid)
            eid = _edge_id(ent_nid, "PARTY_TO", doc_nid)
            if eid in seen:
                continue
            seen.add(eid)
            edges.append({
                "edgeId": eid,
                "conversationId": conversation_id,
                "fromNodeId": ent_nid,
                "toNodeId": doc_nid,
                "edgeType": "PARTY_TO",
                "provenance": "EXTRACTED",
                "confidence": 1.0,
                "sourceFileId": fid,
                "sourceSectionId": None,
                "detail": f"{g['label']} is a party to this document",
            })
    return edges


def _build_llm_section_edges(
    file_id: str,
    file_name: str,
    conversation_id: str,
    sections: List[dict],
) -> List[dict]:
    """LLM-extracted semantic edges — CONDITIONS, SUPERSEDES, EXCEPTIONS, etc."""
    _VALID_TYPES = {
        "CONDITIONS", "SUPERSEDES", "EXCEPTIONS",
        "DEFINES", "OBLIGATES", "PERMITS", "PROHIBITS",
    }
    _VALID_PROV = {"EXTRACTED", "INFERRED", "AMBIGUOUS"}
    _BATCH = 20

    sec_index = {
        s["sectionId"]: _node_id("section", file_id, s["sectionId"])
        for s in sections
    }

    edges = []
    seen: Set[str] = set()

    for i in range(0, len(sections), _BATCH):
        batch = sections[i: i + _BATCH]
        for rel in _extract_clause_relationships(file_id, file_name, batch):
            from_sec = str(rel.get("from_section", "")).strip()
            to_sec = str(rel.get("to_section", "")).strip()
            etype = str(rel.get("edge_type", "")).upper().strip()
            prov = str(rel.get("provenance", "INFERRED")).upper().strip()
            conf = float(rel.get("confidence", 0.7))
            detail = str(rel.get("detail", ""))

            if etype not in _VALID_TYPES:
                continue
            if prov not in _VALID_PROV:
                prov = "AMBIGUOUS"
            if conf < 0.5:
                continue
            if from_sec not in sec_index or to_sec not in sec_index:
                continue

            from_id = sec_index[from_sec]
            to_id = sec_index[to_sec]
            eid = _edge_id(from_id, etype, to_id)
            if eid in seen:
                continue
            seen.add(eid)
            edges.append({
                "edgeId": eid,
                "conversationId": conversation_id,
                "fromNodeId": from_id,
                "toNodeId": to_id,
                "edgeType": etype,
                "provenance": prov,
                "confidence": conf,
                "sourceFileId": file_id,
                "sourceSectionId": from_sec,
                "detail": detail,
            })

    return edges


# ---------------------------------------------------------------------------
# Public entry point — extraction
# ---------------------------------------------------------------------------

def extract_graph(file_id: str, conversation_id: str) -> dict:
    """
    Build and persist the knowledge graph for one file.
    Idempotent — safe to re-run (upserts).

    Returns {nodes, edges, llm_edges} count summary.
    """
    from src.utils.connection_utils import db

    rec = db["filePages"].find_one({"fileId": file_id}, {"fileName": 1})
    if not rec:
        logger.warning("[GraphExtractor] No filePages for fileId=%s", file_id)
        return {}

    file_name = rec.get("fileName", file_id)
    logger.info("[GraphExtractor] Extracting graph for %s", file_name)

    # 1. Load sections
    sections = list(db["fileSections"].find(
        {"fileId": file_id, "conversationId": conversation_id},
        {"_id": 0, "sectionId": 1, "sectionTitle": 1, "content": 1,
         "clauseType": 1, "parentSectionId": 1, "riskLevel": 1, "pageNumber": 1},
    ))
    if not sections:
        logger.warning("[GraphExtractor] No sections for %s — skipping", file_name)
        return {}

    known_sids: Set[str] = {s["sectionId"] for s in sections}

    # 2. Deduplicate entities across conversation
    entity_map = _deduplicate_entities(conversation_id)

    # 3. Build nodes
    doc_node = _build_document_node(file_id, conversation_id)
    sec_nodes = _build_section_nodes(file_id, conversation_id, sections)
    ent_nodes = _build_entity_nodes(entity_map, conversation_id)
    all_nodes = ([doc_node] if doc_node else []) + sec_nodes + ent_nodes

    # 4. Build edges
    structural = _build_structural_edges(file_id, conversation_id, sections)
    crossrefs = _build_crossref_edges(file_id, conversation_id, known_sids)
    doc_rels = _build_doc_relationship_edges(conversation_id)
    entity_edges = _build_entity_doc_edges(entity_map, conversation_id)
    llm_edges = _build_llm_section_edges(file_id, file_name, conversation_id, sections)
    all_edges = structural + crossrefs + doc_rels + entity_edges + llm_edges

    # 5. Upsert to MongoDB
    from pymongo import UpdateOne

    if all_nodes:
        ops = [
            UpdateOne(
                {"nodeId": n["nodeId"], "conversationId": conversation_id},
                {"$set": n},
                upsert=True,
            )
            for n in all_nodes if n.get("nodeId")
        ]
        if ops:
            db["graphNodes"].bulk_write(ops, ordered=False)

    if all_edges:
        ops = [
            UpdateOne(
                {"edgeId": e["edgeId"], "conversationId": conversation_id},
                {"$set": e},
                upsert=True,
            )
            for e in all_edges if e.get("edgeId")
        ]
        if ops:
            db["graphEdges"].bulk_write(ops, ordered=False)

    # Ensure indexes exist
    try:
        db["graphNodes"].create_index(
            [("conversationId", 1), ("nodeType", 1)], background=True
        )
        db["graphNodes"].create_index([("nodeId", 1)], background=True)
        db["graphEdges"].create_index(
            [("conversationId", 1), ("edgeType", 1)], background=True
        )
        db["graphEdges"].create_index(
            [("fromNodeId", 1), ("toNodeId", 1)], background=True
        )
    except Exception:
        pass

    summary = {
        "nodes": len(all_nodes),
        "edges": len(all_edges),
        "llm_edges": len(llm_edges),
    }
    logger.info(
        "[GraphExtractor] %s: %d nodes, %d edges (%d LLM-extracted)",
        file_name, summary["nodes"], summary["edges"], summary["llm_edges"],
    )
    return summary


# ---------------------------------------------------------------------------
# RAG integration helpers
# ---------------------------------------------------------------------------

def resolve_entity_aliases(conversation_id: str, query: str) -> List[str]:
    """
    Given a query, find all entity aliases that match any token in the query.
    Returns expanded list of alias strings to use in broadened searches.

    e.g. query "Mainstay HK obligations" → ["Mainstay Asia Ltd", "Mainstay HK",
                                             "Mainstay (HK) Limited"]
    """
    from src.utils.connection_utils import db

    query_lower = query.lower()
    query_tokens = [t for t in re.split(r"\W+", query_lower) if len(t) >= 3]
    if not query_tokens:
        return []

    # Load all entity nodes for this conversation
    entity_nodes = list(db["graphNodes"].find(
        {"conversationId": conversation_id, "nodeType": {"$in": ["person", "organization"]}},
        {"label": 1, "aliases": 1},
    ))

    expanded: Set[str] = set()
    for node in entity_nodes:
        all_names = [node.get("label", "")] + (node.get("aliases") or [])
        # Check if any alias token matches any query token
        matched = False
        for name in all_names:
            name_tokens = [t for t in re.split(r"\W+", name.lower()) if len(t) >= 3]
            if any(qt in name_tokens or nt in query_tokens
                   for qt in query_tokens for nt in name_tokens
                   if qt == nt):
                matched = True
                break
        if matched:
            for name in all_names:
                if name:
                    expanded.add(name)

    return sorted(expanded)


def get_graph_context_for_sections(
    conversation_id: str,
    seed_section_node_ids: List[str],
    edge_types: Optional[List[str]] = None,
) -> List[dict]:
    """
    Given a list of section nodeIds already in the candidate set, walk the graph
    to find sections connected via semantic edges (CONDITIONS, SUPERSEDES, EXCEPTIONS
    by default) and return their fileSections records.

    Used in Phase 2 to inject clauses that are structurally linked to seed sections
    even when they wouldn't score well in semantic search.
    """
    from src.utils.connection_utils import db

    if not seed_section_node_ids:
        return []

    if edge_types is None:
        edge_types = ["CONDITIONS", "SUPERSEDES", "EXCEPTIONS", "DEFINES"]

    # Find all edges from/to seed nodes with the desired types
    edges = list(db["graphEdges"].find({
        "conversationId": conversation_id,
        "edgeType": {"$in": edge_types},
        "$or": [
            {"fromNodeId": {"$in": seed_section_node_ids}},
            {"toNodeId": {"$in": seed_section_node_ids}},
        ],
    }, {"fromNodeId": 1, "toNodeId": 1, "edgeType": 1, "sourceFileId": 1,
        "sourceSectionId": 1, "detail": 1, "confidence": 1}))

    if not edges:
        return []

    # Collect all connected node IDs
    connected_node_ids: Set[str] = set()
    for e in edges:
        connected_node_ids.add(e["fromNodeId"])
        connected_node_ids.add(e["toNodeId"])
    # Remove seeds — we only want what was NOT already in the candidate set
    new_node_ids = connected_node_ids - set(seed_section_node_ids)
    if not new_node_ids:
        return []

    # Resolve node IDs → (fileId, sectionId) pairs
    graph_nodes = list(db["graphNodes"].find(
        {"nodeId": {"$in": list(new_node_ids)}, "nodeType": "section"},
        {"fileId": 1, "sectionId": 1, "nodeId": 1},
    ))
    if not graph_nodes:
        return []

    # Build lookup for edge detail (so we can tag the injected section)
    node_edge_detail: Dict[str, str] = {}
    for e in edges:
        for nid in (e["fromNodeId"], e["toNodeId"]):
            if nid in new_node_ids and nid not in node_edge_detail:
                node_edge_detail[nid] = f"[GRAPH:{e['edgeType']}] {e.get('detail', '')}"

    # Fetch actual fileSections documents
    or_clauses = [
        {"fileId": n["fileId"], "sectionId": n["sectionId"]}
        for n in graph_nodes
        if n.get("fileId") and n.get("sectionId")
    ]
    if not or_clauses:
        return []

    sections = list(db["fileSections"].find(
        {"conversationId": conversation_id, "$or": or_clauses},
        {"_id": 0},
    ))

    # Build nodeId lookup to tag sections
    node_lookup = {
        (_node_id("section", n["fileId"], n["sectionId"])): n
        for n in graph_nodes
        if n.get("fileId") and n.get("sectionId")
    }
    for sec in sections:
        nid = _node_id("section", sec.get("fileId", ""), sec.get("sectionId", ""))
        sec["_graph_injection"] = True
        sec["_graph_edge_detail"] = node_edge_detail.get(nid, "")

    return sections


# ---------------------------------------------------------------------------
# Visualization helpers — React Flow shaped
# ---------------------------------------------------------------------------

_NODE_COLORS = {
    "document": {
        "master_agreement": "#3B82F6",
        "transaction": "#10B981",
        "modification": "#F59E0B",
        "termination": "#EF4444",
        "standalone": "#6B7280",
    },
    "person": "#8B5CF6",
    "organization": "#EC4899",
}

_SECTION_CLAUSE_COLORS = {
    "payment_terms": "#FEF3C7",
    "liability": "#FEE2E2",
    "termination": "#FCE7F3",
    "confidentiality": "#EDE9FE",
    "governing_law": "#DBEAFE",
    "definitions": "#D1FAE5",
    "order_of_precedence": "#CFFAFE",
    "service_levels": "#ECFDF5",
}

_EDGE_COLORS = {
    "CONTAINS": "#9CA3AF",
    "CHILD_OF": "#D1D5DB",
    "REFERENCES": "#60A5FA",
    "CONDITIONS": "#F59E0B",
    "SUPERSEDES": "#EF4444",
    "EXCEPTIONS": "#F97316",
    "DEFINES": "#10B981",
    "OBLIGATES": "#8B5CF6",
    "PERMITS": "#06B6D4",
    "PROHIBITS": "#EC4899",
    "PARTY_TO": "#A78BFA",
    "MASTER_OF": "#3B82F6",
    "NOVATES": "#F59E0B",
    "TERMINATES": "#EF4444",
    "RENEWS": "#10B981",
}


def get_graph_for_conversation(conversation_id: str) -> dict:
    """
    High-level graph: Document + Person + Org nodes only.
    Suitable for the overview visualization.
    Returns React-Flow-compatible {nodes, edges, stats}.

    Document-to-document edges (MASTER_OF, NOVATES, TERMINATES, RENEWS) are
    always fetched live from documentRelationships so the graph stays current
    even if the graph was built before relationships were detected.
    """
    from src.utils.connection_utils import db
    from pymongo import UpdateOne

    top_types = {"document", "person", "organization"}
    top_skip_edges = {"CONTAINS", "CHILD_OF"}

    raw_nodes = {
        n["nodeId"]: n
        for n in db["graphNodes"].find(
            {"conversationId": conversation_id, "nodeType": {"$in": list(top_types)}},
            {"_id": 0},
        )
    }

    # Always rebuild document-to-document edges from live documentRelationships.
    # This ensures MASTER_OF / NOVATES / TERMINATES / RENEWS are present even if
    # the graph was originally built before build_document_relationships() ran.
    live_doc_edges = _build_doc_relationship_edges(conversation_id)
    if live_doc_edges:
        ops = [
            UpdateOne(
                {"edgeId": e["edgeId"], "conversationId": conversation_id},
                {"$set": e},
                upsert=True,
            )
            for e in live_doc_edges
        ]
        db["graphEdges"].bulk_write(ops, ordered=False)

    raw_edges = [
        e for e in db["graphEdges"].find(
            {"conversationId": conversation_id,
             "edgeType": {"$nin": list(top_skip_edges)}},
            {"_id": 0},
        )
        if e.get("fromNodeId") in raw_nodes and e.get("toNodeId") in raw_nodes
    ]

    rf_nodes = []
    for nid, n in raw_nodes.items():
        nt = n.get("nodeType", "document")
        if nt == "document":
            role = (n.get("properties") or {}).get("functionalRole", "standalone")
            color = _NODE_COLORS["document"].get(role, "#6B7280")
        else:
            color = _NODE_COLORS.get(nt, "#6B7280")

        rf_nodes.append({
            "id": nid,
            "type": "contractNode",
            "data": {
                "label": n.get("label", nid),
                "nodeType": nt,
                "aliases": n.get("aliases", []),
                "color": color,
                "properties": n.get("properties", {}),
                "fileId": n.get("fileId"),
            },
            "position": {"x": 0, "y": 0},
        })

    seen_eids: Set[str] = set()
    rf_edges = []
    for e in raw_edges:
        eid = e.get("edgeId", "")
        if eid in seen_eids:
            continue
        seen_eids.add(eid)
        etype = e.get("edgeType", "")
        rf_edges.append({
            "id": eid,
            "source": e.get("fromNodeId"),
            "target": e.get("toNodeId"),
            "label": etype,
            "style": {"stroke": _EDGE_COLORS.get(etype, "#9CA3AF")},
            "data": {
                "edgeType": etype,
                "provenance": e.get("provenance", "EXTRACTED"),
                "confidence": e.get("confidence", 1.0),
                "detail": e.get("detail", ""),
            },
        })

    total_nodes = db["graphNodes"].count_documents({"conversationId": conversation_id})
    total_edges = db["graphEdges"].count_documents({"conversationId": conversation_id})

    return {
        "nodes": rf_nodes,
        "edges": rf_edges,
        "stats": {
            "totalNodes": total_nodes,
            "totalEdges": total_edges,
            "visibleNodes": len(rf_nodes),
            "visibleEdges": len(rf_edges),
        },
    }


def get_section_graph(file_id: str, conversation_id: str) -> dict:
    """
    Drill-down graph for one document — section nodes + semantic edges.
    Returns React-Flow-compatible {nodes, edges}.
    """
    from src.utils.connection_utils import db

    # All nodes belonging to this file
    file_nodes = {
        n["nodeId"]: n
        for n in db["graphNodes"].find(
            {"conversationId": conversation_id, "fileId": file_id},
            {"_id": 0},
        )
    }
    # Entity nodes that are PARTY_TO this file
    doc_nid = _node_id("document", file_id)
    party_edges = list(db["graphEdges"].find({
        "conversationId": conversation_id,
        "edgeType": "PARTY_TO",
        "toNodeId": doc_nid,
    }, {"fromNodeId": 1}))
    party_nids = {e["fromNodeId"] for e in party_edges}
    entity_nodes = {
        n["nodeId"]: n
        for n in db["graphNodes"].find(
            {"nodeId": {"$in": list(party_nids)}},
            {"_id": 0},
        )
    }
    all_nodes = {**file_nodes, **entity_nodes}

    # Edges where both endpoints are in our node set
    raw_edges = list(db["graphEdges"].find(
        {
            "conversationId": conversation_id,
            "$or": [
                {"fromNodeId": {"$in": list(all_nodes)}},
                {"toNodeId": {"$in": list(all_nodes)}},
            ],
        },
        {"_id": 0},
    ))

    rf_nodes = []
    for nid, n in all_nodes.items():
        nt = n.get("nodeType", "section")
        if nt == "document":
            role = (n.get("properties") or {}).get("functionalRole", "standalone")
            color = _NODE_COLORS["document"].get(role, "#6B7280")
        elif nt in ("person", "organization"):
            color = _NODE_COLORS.get(nt, "#A78BFA")
        else:
            ct = (n.get("properties") or {}).get("clauseType", "")
            color = _SECTION_CLAUSE_COLORS.get(ct, "#F3F4F6")

        rf_nodes.append({
            "id": nid,
            "type": "contractNode",
            "data": {
                "label": n.get("label", nid),
                "nodeType": nt,
                "aliases": n.get("aliases", []),
                "color": color,
                "properties": n.get("properties", {}),
                "fileId": n.get("fileId"),
                "sectionId": n.get("sectionId"),
            },
            "position": {"x": 0, "y": 0},
        })

    seen_eids: Set[str] = set()
    rf_edges = []
    for e in raw_edges:
        eid = e.get("edgeId", "")
        if eid in seen_eids:
            continue
        # Both endpoints must be visible
        if e.get("fromNodeId") not in all_nodes or e.get("toNodeId") not in all_nodes:
            continue
        seen_eids.add(eid)
        etype = e.get("edgeType", "")
        rf_edges.append({
            "id": eid,
            "source": e.get("fromNodeId"),
            "target": e.get("toNodeId"),
            "label": etype,
            "animated": etype in ("CONDITIONS", "SUPERSEDES", "EXCEPTIONS"),
            "style": {"stroke": _EDGE_COLORS.get(etype, "#9CA3AF")},
            "data": {
                "edgeType": etype,
                "provenance": e.get("provenance", "EXTRACTED"),
                "confidence": e.get("confidence", 1.0),
                "detail": e.get("detail", ""),
            },
        })

    return {"nodes": rf_nodes, "edges": rf_edges}
