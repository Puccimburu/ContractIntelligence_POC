"""
Document Relationship Service

Detects and stores cross-document relationships within a conversation's
uploaded corpus. Relationships captured:

  novation      — Novation Agreement names an underlying agreement being transferred.
                  Links: novation → original agreement (parent); outgoing party → incoming party.

  termination   — Termination Letter names a specific Work Order or agreement.
                  Links: termination letter → the terminated agreement.

  renewal       — Multiple Work Orders for the same consultant are detected as
                  successive renewals (ordered by commencement date).
                  Links: each WO → its predecessor.

  master_child  — Every Work Order references a Master Agreement.
                  Links: work order → master agreement.

Results are written to the `documentRelationships` collection:
{
    "conversationId": str,
    "fromFileId":     str,   # the document making the reference
    "toFileId":       str,   # the document being referenced (None if unresolved)
    "relationshipType": str, # novation | termination | renewal | master_child
    "fromDocumentType": str,
    "toDocumentType":   str,
    "detail":           str, # human-readable description of the link
    "resolved":         bool,
}

Entry points:

    build_document_relationships(conversation_id)
        Full build — run after all documents in a conversation have been
        classified and had metadata extracted.

    get_relationships_for_file(conversation_id, file_id)
        Returns all relationships where fromFileId or toFileId == file_id.

    get_consultant_work_order_chain(conversation_id, consultant_name)
        Returns the ordered chain of Work Orders for a consultant, with
        each entry annotated with its predecessor and whether it is current.
"""

import re
import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from src.utils.connection_utils import db

logger = logging.getLogger(__name__)

_COLL = "documentRelationships"

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches a named agreement reference: "Master Consultancy Agreement dated …",
# "Work Order dated 31 January 2023 for [Consultant]", etc.
_NAMED_AGREEMENT_RE = re.compile(
    r'(master\s+consultancy\s+agreement|master\s+services\s+agreement'
    r'|work\s+order(?:\s+dated\s+[\d\w\s,]+)?(?:\s+for\s+[\w\s]+)?'
    r'|novation\s+agreement|termination\s+letter)',
    re.IGNORECASE,
)

_DATE_IN_TEXT_RE = re.compile(
    r'\b(\d{1,2}(?:st|nd|rd|th)?\s+'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_iso_date(date_str: Optional[str]) -> Optional[date]:
    """Parse a date string (ISO or common formats) to a date object."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str[:10], fmt).date()
        except ValueError:
            continue
    return None


def _get_extraction_value(file_id: str, field_name: str) -> Optional[str]:
    """Fetch a single extracted attribute value from documentextractions."""
    doc = db["documentextractions"].find_one(
        {"fileId": file_id, "name": field_name},
        {"value": 1},
    )
    return doc.get("value") if doc else None


def _get_all_extractions(file_id: str) -> Dict[str, str]:
    """Return all extraction name→value pairs for a file."""
    return {
        d["name"]: d.get("value", "")
        for d in db["documentextractions"].find({"fileId": file_id}, {"name": 1, "value": 1})
        if d.get("name") and d.get("value") not in (None, "Not Found", "")
    }


def _upsert_relationship(rel: Dict[str, Any]) -> None:
    db[_COLL].update_one(
        {
            "conversationId": rel["conversationId"],
            "fromFileId": rel["fromFileId"],
            "toFileId": rel.get("toFileId"),
            "relationshipType": rel["relationshipType"],
        },
        {"$set": rel},
        upsert=True,
    )


def _files_for_conversation(conversation_id: str) -> List[Dict]:
    """Return all filePages records for the conversation with classification metadata."""
    return list(db["filePages"].find(
        {"conversationId": conversation_id},
        {"fileId": 1, "fileName": 1, "documentType": 1, "documentRank": 1,
         "effectiveDate": 1, "functionalRole": 1},
    ))


def _functional_role(file_dict: Dict) -> str:
    """
    Return the functionalRole for a file, falling back to a derivation from
    documentType so the service works even before entity extraction runs.
    """
    role = file_dict.get("functionalRole") or ""
    if role in ("master_agreement", "transaction", "modification", "termination", "standalone"):
        return role

    # Derive from documentType as best-effort fallback
    dt = (file_dict.get("documentType") or "").lower()
    if dt in ("framework", "master consultancy agreement", "master services agreement",
              "master_services_agreement", "framework agreement"):
        return "master_agreement"
    if dt in ("work_order", "work order", "sow", "statement of work", "purchase order"):
        return "transaction"
    if dt in ("novation", "novation agreement", "amendment", "addendum", "side letter",
              "variation", "modification"):
        return "modification"
    if dt in ("termination_notice", "termination letter", "termination", "notice"):
        return "termination"
    return "standalone"


def _entity_persons_for_file(file_id: str) -> List[str]:
    """Return all normalized person entity keys extracted from this file."""
    return [
        r.get("entityKey", "")
        for r in db["entityIndex"].find(
            {"fileId": file_id, "entityType": "person"},
            {"entityKey": 1},
        )
        if r.get("entityKey")
    ]


def _entity_refs_for_file(file_id: str) -> List[str]:
    """Return all agreement reference values extracted from this file."""
    return [
        r.get("entityValue", "")
        for r in db["entityIndex"].find(
            {"fileId": file_id, "entityType": "agreement_ref"},
            {"entityValue": 1},
        )
        if r.get("entityValue")
    ]


def _files_sharing_person(
    conversation_id: str, person_key: str, exclude_file_id: str
) -> List[str]:
    """Return fileIds in the conversation that share this person entity key."""
    return [
        r["fileId"]
        for r in db["entityIndex"].find(
            {
                "conversationId": conversation_id,
                "entityType": "person",
                "entityKey": person_key,
                "fileId": {"$ne": exclude_file_id},
            },
            {"fileId": 1},
        )
        if r.get("fileId")
    ]


def _file_name_similarity(a: str, b: str) -> float:
    """Rough similarity between two file name tokens (for fuzzy matching)."""
    a_tokens = set(re.sub(r'[^a-z0-9]', ' ', a.lower()).split())
    b_tokens = set(re.sub(r'[^a-z0-9]', ' ', b.lower()).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _find_file_by_name_fragment(files: List[Dict], fragment: str) -> Optional[Dict]:
    """Return the file whose name best matches the given fragment (>= 0.25 similarity)."""
    if not fragment or not files:
        return None
    scored = [
        (f, _file_name_similarity(f.get("fileName", ""), fragment))
        for f in files
    ]
    scored.sort(key=lambda x: -x[1])
    best_file, best_score = scored[0]
    return best_file if best_score >= 0.25 else None


# ---------------------------------------------------------------------------
# Relationship detectors
# ---------------------------------------------------------------------------

def _get_identifier_keys_for_file(file_id: str) -> List[str]:
    """Return all identifier values (normalized) from documentEntities for a file."""
    doc = db["documentEntities"].find_one({"fileId": file_id}, {"identifiers": 1})
    if not doc:
        return []
    result = []
    for ident in doc.get("identifiers", []):
        v = (ident.get("value") or "").strip()
        if v:
            result.append(re.sub(r'[^a-z0-9]', '', v.lower()))
    return result


def _find_master_by_identifier(masters: List[Dict], ref_text: str) -> Optional[Dict]:
    """
    Try to resolve a master by matching ref_text against identifier values stored
    in documentEntities. Handles contract reference codes like "Aston-22-7464" that
    don't appear in file names but do appear as document identifiers.
    """
    normalized_ref = re.sub(r'[^a-z0-9]', '', ref_text.lower())
    if not normalized_ref or len(normalized_ref) < 4:
        return None
    for master in masters:
        for ik in _get_identifier_keys_for_file(master["fileId"]):
            if normalized_ref in ik or ik in normalized_ref:
                return master
    return None


def _detect_master_child(conversation_id: str, files: List[Dict]) -> None:
    """
    Link every transaction and modification (addendum) document to its master agreement.

    Covers:
    - Work Orders / SOWs / POs (transaction role)
    - Addenda / SaaS Schedules / Amendments (modification role) that depend on a master

    Detection strategy — four layers:
    1. Entity index: agreement_ref entities matched against master file names.
    2. Identifier match: agreement_ref matched against document identifier codes stored
       in documentEntities (e.g. contract numbers like "Aston-22-7464").
    3. Extraction fallback: "Master Agreement Reference" prompt value.
    4. Uniqueness fallback: if only one master_agreement exists, link all
       unresolved transactions/modifications to it.
    """
    masters = [f for f in files if _functional_role(f) == "master_agreement"]
    dependents = [f for f in files if _functional_role(f) in ("transaction", "modification")]

    if not masters or not dependents:
        return

    for tx in dependents:
        target = None

        # Layer 1: entity refs matched against file name
        refs = _entity_refs_for_file(tx["fileId"])
        for ref_text in refs:
            candidate = _find_file_by_name_fragment(masters, ref_text)
            if candidate:
                target = candidate
                break

        # Layer 2: entity refs matched against document identifier codes
        if not target:
            for ref_text in refs:
                candidate = _find_master_by_identifier(masters, ref_text)
                if candidate:
                    target = candidate
                    break

        # Layer 3: extraction fallback
        if not target:
            ref_text = _get_extraction_value(tx["fileId"], "Master Agreement Reference") or ""
            if ref_text:
                target = _find_file_by_name_fragment(masters, ref_text) or \
                         _find_master_by_identifier(masters, ref_text)

        # Layer 4: single master fallback
        if not target and len(masters) == 1:
            target = masters[0]

        _upsert_relationship({
            "conversationId": conversation_id,
            "fromFileId": tx["fileId"],
            "toFileId": target["fileId"] if target else None,
            "relationshipType": "master_child",
            "fromDocumentType": tx.get("documentType") or _functional_role(tx),
            "toDocumentType": target.get("documentType", "master_agreement") if target else "master_agreement",
            "detail": (
                f"{tx.get('fileName','document')} is issued under "
                f"{target.get('fileName','[unresolved master]') if target else '[unresolved master]'}"
            ),
            "resolved": target is not None,
        })
        logger.info("master_child: %s → %s", tx.get("fileName"), target.get("fileName") if target else "unresolved")


def _detect_novation_links(conversation_id: str, files: List[Dict]) -> None:
    """
    Link each modification document (Novation / Amendment / Side Letter) to
    the agreements it affects.

    Detection strategy:
    1. Entity index: agreement_ref entities in the modification doc matched
       against other files in the conversation.
    2. Shared-person overlap: persons in the modification doc also appear in
       transaction docs → those transactions are candidates being novated.
    3. Extraction fallback: "Novated Agreements" prompt value.
    """
    modifications = [f for f in files if _functional_role(f) == "modification"]

    for mod in modifications:
        # Skip modifications that already have a resolved master_child link —
        # those are addenda handled by _detect_master_child (e.g. SaaS Agreement).
        already_linked = db[_COLL].find_one({
            "conversationId": conversation_id,
            "fromFileId": mod["fileId"],
            "relationshipType": "master_child",
            "resolved": True,
        })
        if already_linked:
            continue

        linked_any = False

        # Layer 1: entity agreement refs
        refs = _entity_refs_for_file(mod["fileId"])
        for ref_text in refs:
            target = _find_file_by_name_fragment(files, ref_text)
            if target and target["fileId"] != mod["fileId"]:
                _upsert_relationship({
                    "conversationId": conversation_id,
                    "fromFileId": mod["fileId"],
                    "toFileId": target["fileId"],
                    "relationshipType": "novation",
                    "fromDocumentType": mod.get("documentType") or _functional_role(mod),
                    "toDocumentType": target.get("documentType") or _functional_role(target),
                    "detail": f"Modifies/novates: {ref_text[:120]}",
                    "resolved": True,
                })
                linked_any = True

        # Layer 2: persons shared with transaction docs
        person_keys = _entity_persons_for_file(mod["fileId"])
        for pkey in person_keys:
            shared_ids = _files_sharing_person(conversation_id, pkey, mod["fileId"])
            for shared_id in shared_ids:
                shared_file = next((f for f in files if f["fileId"] == shared_id), None)
                if not shared_file:
                    continue
                # Only link to transactions (Work Orders) to avoid false positives
                if _functional_role(shared_file) != "transaction":
                    continue
                _upsert_relationship({
                    "conversationId": conversation_id,
                    "fromFileId": mod["fileId"],
                    "toFileId": shared_id,
                    "relationshipType": "novation",
                    "fromDocumentType": mod.get("documentType") or _functional_role(mod),
                    "toDocumentType": shared_file.get("documentType") or "transaction",
                    "detail": f"Shares consultant '{pkey}' with transaction document",
                    "resolved": True,
                })
                linked_any = True

        # Layer 3: extraction fallback
        if not linked_any:
            extractions = _get_all_extractions(mod["fileId"])
            novated_text = extractions.get("Novated Agreements", "")
            fragments = [s.strip() for s in novated_text.split(";") if s.strip()] if novated_text else []
            for fragment in fragments:
                target = _find_file_by_name_fragment(files, fragment)
                _upsert_relationship({
                    "conversationId": conversation_id,
                    "fromFileId": mod["fileId"],
                    "toFileId": target["fileId"] if target else None,
                    "relationshipType": "novation",
                    "fromDocumentType": mod.get("documentType") or _functional_role(mod),
                    "toDocumentType": target.get("documentType", "Unknown") if target else "Unknown",
                    "detail": f"Modifies/novates: {fragment[:120]}",
                    "resolved": target is not None,
                })
                if target:
                    linked_any = True

        if not linked_any:
            _upsert_relationship({
                "conversationId": conversation_id,
                "fromFileId": mod["fileId"],
                "toFileId": None,
                "relationshipType": "novation",
                "fromDocumentType": mod.get("documentType") or _functional_role(mod),
                "toDocumentType": "Unknown",
                "detail": "Modified agreements could not be resolved",
                "resolved": False,
            })


def _detect_termination_links(conversation_id: str, files: List[Dict]) -> None:
    """
    Link each termination document to the agreement it ends.

    Detection strategy:
    1. Entity index: persons shared between the termination doc and transaction docs.
    2. Entity index: agreement_ref entities matched against file names.
    3. Extraction fallback: "Agreement Being Terminated" prompt value.
    """
    terminations = [f for f in files if _functional_role(f) == "termination"]

    for term in terminations:
        linked_any = False

        # Layer 1: shared persons → find transaction docs they appear in
        person_keys = _entity_persons_for_file(term["fileId"])
        for pkey in person_keys:
            shared_ids = _files_sharing_person(conversation_id, pkey, term["fileId"])
            for shared_id in shared_ids:
                shared_file = next((f for f in files if f["fileId"] == shared_id), None)
                if not shared_file or _functional_role(shared_file) != "transaction":
                    continue
                _upsert_relationship({
                    "conversationId": conversation_id,
                    "fromFileId": term["fileId"],
                    "toFileId": shared_id,
                    "relationshipType": "termination",
                    "fromDocumentType": term.get("documentType") or "termination",
                    "toDocumentType": shared_file.get("documentType") or "transaction",
                    "detail": f"Terminates Work Order for consultant '{pkey}'",
                    "resolved": True,
                })
                linked_any = True

        # Layer 2: agreement ref entities
        if not linked_any:
            for ref_text in _entity_refs_for_file(term["fileId"]):
                target = _find_file_by_name_fragment(files, ref_text)
                if target and target["fileId"] != term["fileId"]:
                    _upsert_relationship({
                        "conversationId": conversation_id,
                        "fromFileId": term["fileId"],
                        "toFileId": target["fileId"],
                        "relationshipType": "termination",
                        "fromDocumentType": term.get("documentType") or "termination",
                        "toDocumentType": target.get("documentType") or _functional_role(target),
                        "detail": f"Terminates: {ref_text[:120]}",
                        "resolved": True,
                    })
                    linked_any = True

        # Layer 3: extraction fallback
        if not linked_any:
            terminated_text = _get_extraction_value(term["fileId"], "Agreement Being Terminated") or ""
            target = _find_file_by_name_fragment(files, terminated_text) if terminated_text else None
            _upsert_relationship({
                "conversationId": conversation_id,
                "fromFileId": term["fileId"],
                "toFileId": target["fileId"] if target else None,
                "relationshipType": "termination",
                "fromDocumentType": term.get("documentType") or "termination",
                "toDocumentType": target.get("documentType", "Unknown") if target else "Unknown",
                "detail": f"Terminates: {terminated_text[:120] or '[unresolved]'}",
                "resolved": target is not None,
            })

        logger.info("termination: %s → linked_any=%s", term.get("fileName"), linked_any)


def _detect_renewal_chains(conversation_id: str, files: List[Dict]) -> None:
    """
    Group transaction documents by person entity, sort by effective date,
    link each to its predecessor as a renewal.

    Uses entity index persons — no dependency on "Consultant Name" extraction prompt.
    Works for any company format: Work Order, SOW, Task Order, PO, etc.
    """
    transactions = [f for f in files if _functional_role(f) == "transaction"]
    if not transactions:
        return

    tx_ids = [f["fileId"] for f in transactions]
    tx_map = {f["fileId"]: f for f in transactions}

    # person_key → list of fileIds that are transaction docs and share this person
    person_to_txs: Dict[str, List[str]] = {}
    for rec in db["entityIndex"].find(
        {"conversationId": conversation_id, "entityType": "person",
         "fileId": {"$in": tx_ids}},
        {"entityKey": 1, "fileId": 1},
    ):
        pkey = rec.get("entityKey", "")
        fid = rec.get("fileId", "")
        if pkey and fid:
            person_to_txs.setdefault(pkey, [])
            if fid not in person_to_txs[pkey]:
                person_to_txs[pkey].append(fid)

    # Also check documentextractions "Consultant Name" as supplement
    for tx in transactions:
        name = _get_extraction_value(tx["fileId"], "Consultant Name")
        if name and name != "Not Found":
            normalised = re.sub(r'\s+', ' ', name.strip()).lower()
            person_to_txs.setdefault(normalised, [])
            if tx["fileId"] not in person_to_txs[normalised]:
                person_to_txs[normalised].append(tx["fileId"])

    for person_key, file_ids in person_to_txs.items():
        if len(file_ids) < 2:
            continue  # single doc — no renewal chain

        # Sort by effective date (ascending); unknown dates last
        def _sort_date(fid: str) -> tuple:
            d = tx_map.get(fid, {}).get("effectiveDate") or ""
            return (not bool(d), d)

        sorted_ids = sorted(file_ids, key=_sort_date)

        for i in range(1, len(sorted_ids)):
            curr_id = sorted_ids[i]
            prev_id = sorted_ids[i - 1]
            curr = tx_map.get(curr_id, {})
            prev = tx_map.get(prev_id, {})

            _upsert_relationship({
                "conversationId": conversation_id,
                "fromFileId": curr_id,
                "toFileId": prev_id,
                "relationshipType": "renewal",
                "fromDocumentType": curr.get("documentType") or "transaction",
                "toDocumentType": prev.get("documentType") or "transaction",
                "detail": (
                    f"Transaction for '{person_key}' renews predecessor "
                    f"(prev: {prev.get('effectiveDate','?')}, this: {curr.get('effectiveDate','?')})"
                ),
                "resolved": True,
                "personKey": person_key,
            })
            logger.info("renewal: %s → %s (person: %s)", curr.get("fileName"), prev.get("fileName"), person_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_document_relationships(conversation_id: str) -> Dict[str, int]:
    """
    Full rebuild of all document relationships for a conversation.
    Safe to call multiple times — uses upserts.

    Returns a summary dict: {relationship_type: count_created_or_updated}
    """
    logger.info("build_document_relationships: conversation=%s", conversation_id)
    files = _files_for_conversation(conversation_id)
    if not files:
        logger.warning("No files found for conversation %s", conversation_id)
        return {}

    before_count = db[_COLL].count_documents({"conversationId": conversation_id})

    _detect_master_child(conversation_id, files)
    _detect_novation_links(conversation_id, files)
    _detect_termination_links(conversation_id, files)
    _detect_renewal_chains(conversation_id, files)

    after_count = db[_COLL].count_documents({"conversationId": conversation_id})

    # Summarise by type
    summary = {}
    for rel_type in ("master_child", "novation", "termination", "renewal"):
        summary[rel_type] = db[_COLL].count_documents(
            {"conversationId": conversation_id, "relationshipType": rel_type}
        )
    logger.info("build_document_relationships done: %s relationships total", after_count)
    return summary


def get_relationships_for_file(conversation_id: str, file_id: str) -> List[Dict]:
    """Return all relationships where this file is the source or target."""
    return list(db[_COLL].find(
        {
            "conversationId": conversation_id,
            "$or": [{"fromFileId": file_id}, {"toFileId": file_id}],
        },
        {"_id": 0},
    ))


def get_consultant_work_order_chain(
    conversation_id: str, consultant_name: str
) -> List[Dict]:
    """
    Return the ordered chain of transaction documents for a named person.

    Schema-agnostic: works via entity index (no "Consultant Name" prompt required).
    Falls back to documentextractions for commencement/expiry dates if available.

    Each entry:
    {
        "fileId": str,
        "fileName": str,
        "functionalRole": str,
        "effectiveDate": str | None,
        "commencementDate": str | None,   # from extraction if available
        "expiryDate": str | None,
        "serviceFee": str | None,
        "isCurrent": bool,
        "isTerminated": bool,
        "predecessorFileId": str | None,
        "allDocuments": bool,  # False = transaction only, True = all matching docs
    }
    """
    from src.services.entity_extractor import find_files_by_entity

    normalised_query = re.sub(r'\s+', ' ', consultant_name.strip()).lower()

    # Primary: entity index (schema-agnostic)
    matching_file_ids = find_files_by_entity(conversation_id, consultant_name, entity_type='person')

    # Supplement: documentextractions "Consultant Name" (for older indexed docs)
    extractions = list(db["documentextractions"].find(
        {"name": "Consultant Name"},
        {"fileId": 1, "value": 1},
    ))
    for e in extractions:
        fid = e.get("fileId")
        val = re.sub(r'\s+', ' ', (e.get("value") or "").strip()).lower()
        if val == normalised_query and fid and fid not in matching_file_ids:
            matching_file_ids.append(fid)

    if not matching_file_ids:
        return []

    # Fetch filePages metadata
    pages_map = {
        r["fileId"]: r
        for r in db["filePages"].find(
            {"conversationId": conversation_id, "fileId": {"$in": matching_file_ids}},
            {"fileId": 1, "fileName": 1, "effectiveDate": 1, "functionalRole": 1,
             "documentType": 1, "documentRank": 1},
        )
    }

    # Gather extracted fields per file (best-effort, not required)
    def _field(fid: str, field: str) -> Optional[str]:
        doc = db["documentextractions"].find_one({"fileId": fid, "name": field}, {"value": 1})
        v = doc.get("value") if doc else None
        return None if v in (None, "Not Found", "") else v

    # Also pull financial terms from documentEntities bag-of-entities
    def _fee_from_entities(fid: str) -> Optional[str]:
        ent = db["documentEntities"].find_one({"fileId": fid}, {"financial_terms": 1})
        if not ent:
            return None
        terms = ent.get("financial_terms", [])
        if terms:
            return "; ".join(
                f"{t.get('key','Fee')}: {t.get('value','')} {t.get('currency','') or ''}".strip()
                for t in terms[:3]
            )
        return None

    # Build renewal chain from relationships
    renewal_rels = {
        r["fromFileId"]: r["toFileId"]
        for r in db[_COLL].find(
            {"conversationId": conversation_id, "relationshipType": "renewal",
             "fromFileId": {"$in": matching_file_ids}},
        )
    }

    # Find terminated file IDs
    terminated_ids = set(
        r["toFileId"]
        for r in db[_COLL].find(
            {"conversationId": conversation_id, "relationshipType": "termination",
             "toFileId": {"$in": matching_file_ids}},
        )
        if r.get("toFileId")
    )

    entries = []
    for fid in matching_file_ids:
        page = pages_map.get(fid, {})
        role = _functional_role(page)
        effective = page.get("effectiveDate")
        # Prefer extracted commencement date over effectiveDate for sorting
        start_str = _field(fid, "Commencement Date") or effective

        entries.append({
            "fileId": fid,
            "fileName": page.get("fileName", fid),
            "functionalRole": role,
            "effectiveDate": effective,
            "commencementDate": start_str,
            "expiryDate": _field(fid, "Expiry Date"),
            "serviceFee": _field(fid, "Service Fee") or _fee_from_entities(fid),
            "isCurrent": False,
            "isTerminated": fid in terminated_ids,
            "predecessorFileId": renewal_rels.get(fid),
            "_sort_date": _parse_iso_date(start_str) or date.min,
            "_role_order": {"master_agreement": 0, "transaction": 1,
                            "modification": 2, "termination": 3}.get(role, 4),
        })

    # Sort: transaction docs first, then by date ascending
    entries.sort(key=lambda x: (x["_role_order"], x["_sort_date"]))
    for e in entries:
        del e["_sort_date"]
        del e["_role_order"]

    # Mark the most recent non-terminated transaction as current
    active_transactions = [e for e in entries
                           if e["functionalRole"] == "transaction" and not e["isTerminated"]]
    if active_transactions:
        active_transactions[-1]["isCurrent"] = True

    return entries
