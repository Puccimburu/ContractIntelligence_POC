"""
Section-level retrieval with multi-hop cross-reference expansion.

retrieve_sections(conversation_id, query, top_k=20)
    Returns (context_text, citation_metadata) — same signature as
    load_full_documents_with_citations so the caller needs no changes.

    Phase 1  Semantic entry — Qdrant vector search on contract_sections,
             scoped to the conversation, top_k results.
    Phase 2  Cross-ref expansion — follow one level of resolved cross-refs
             stored in fileCrossRefs; pulls in linked sections from same or
             other documents.
    Phase 3  Cross-encoder re-ranking — local neural model scores every
             (query, section) pair and keeps the top-ranked sections.
             Zero API cost, ~20-50 ms, no rate limits.
    Phase 4  Text assembly — full content of confirmed sections with
             [Source N: filename, Page X] citation markers.

all_sections_embedded(conversation_id)
    Returns True if at least one file in the conversation has
    sections_embedded == True in filePages.
"""
import json
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_SECTION_COLLECTION = "contract_sections"


# ---------------------------------------------------------------------------
# Public readiness check
# ---------------------------------------------------------------------------

def all_sections_embedded(conversation_id: str) -> bool:
    """
    Returns True if any sections have been indexed in Qdrant for this conversation.

    Queries Qdrant directly rather than relying on the filePages.sections_embedded
    flag, which can be missing if embedding completed but the flag write failed or
    if only some files were embedded before a retry.  The section retriever handles
    partial coverage gracefully — it searches only what is in Qdrant.
    """
    from src.utils.qdrant.qdrant_utils import get_qdrant_client
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    import ssl
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

    def _is_transient(exc: BaseException) -> bool:
        if isinstance(exc, ssl.SSLError):
            return True
        msg = str(exc).lower()
        return any(kw in msg for kw in ("sslv3", "bad record mac", "connection reset", "broken pipe", "timed out"))

    try:
        client = get_qdrant_client()

        @retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        def _count():
            return client.count(
                collection_name=_SECTION_COLLECTION,
                count_filter=Filter(must=[
                    FieldCondition(
                        key="conversationId",
                        match=MatchValue(value=conversation_id),
                    )
                ]),
                exact=False,
            )

        result = _count()
        return result.count > 0
    except Exception as e:
        logger.warning(f"[SectionRetriever] all_sections_embedded check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Phase 0: Query expansion
# ---------------------------------------------------------------------------

def _phase0_query_expansion(query: str) -> List[str]:
    """
    Use the LLM to decompose the query into 3–5 targeted sub-queries that
    attack the information need from different angles: synonyms, structural
    headings (e.g. "Schedule 7"), specific value patterns (£ amounts,
    percentages), and related clause types.

    The original query is always the first element.  Falls back to
    [query] if the LLM call fails so the pipeline is never blocked.
    """
    from src.utils.llm_utils import invoke_with_costing_evalution

    prompt = (
        "You are a legal document search specialist. Given a user question about a "
        "contract or legal document, generate 3–5 targeted search queries that cover "
        "different angles: synonyms, structural headings (e.g. 'Schedule 7'), specific "
        "values (£ amounts, percentages), and related clause types.\n\n"
        f"User question: {query}\n\n"
        "Output ONLY a JSON array of short search strings (5–10 words each). "
        "Include the original question as the first element.\n"
        'Example: ["annual licence fee", "Schedule 7 hosting charge £90000", '
        '"SaaS Agreement payment terms"]'
    )
    try:
        response = invoke_with_costing_evalution(prompt=prompt)
        raw = response.content.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            queries = json.loads(raw[start:end])
            if isinstance(queries, list) and queries:
                result = [query] + [q for q in queries if q != query]
                logger.info(
                    "[SectionRetriever] Phase 0: expanded to %d sub-queries", len(result)
                )
                return result[:5]  # cap at 5
    except Exception as e:
        logger.warning("[SectionRetriever] Phase 0 query expansion failed: %s", e)
    return [query]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _build_file_meta_map(conversation_id: str) -> Dict[str, dict]:
    """
    Load per-file temporal and risk metadata from filePages for all files in the
    conversation.  Returns a dict keyed by fileId.

    Fields carried forward to every section dict:
      effectiveDate   — ISO date string extracted at parse time ("2023-03-30")
      documentType    — company-specific type string ("work_order", "framework", …)
      documentRank    — int: 0=framework, 1=amendment, 2=work_order, 3=termination
      functionalRole  — schema-agnostic role: master_agreement | transaction |
                        modification | termination | standalone
      riskFlags       — list of {sectionId, riskLevel, riskNote} for high-risk sections
    """
    # Functional role → precedence rank (lower = more general = lower priority)
    _ROLE_RANK = {
        "master_agreement": 0,
        "standalone": 0,
        "modification": 1,
        "transaction": 2,
        "termination": 3,
    }

    from src.utils.connection_utils import db
    records = db["filePages"].find(
        {"conversationId": conversation_id},
        {"fileId": 1, "effectiveDate": 1, "documentType": 1, "documentRank": 1,
         "functionalRole": 1, "riskFlags": 1},
    )
    result = {}
    for r in records:
        fid = r.get("fileId")
        if not fid:
            continue
        role = r.get("functionalRole") or ""
        # If entity extractor hasn't run yet, derive role from documentRank as fallback
        if role not in ("master_agreement", "transaction", "modification", "termination", "standalone"):
            rank = r.get("documentRank", 1)
            role = {0: "master_agreement", 1: "modification", 2: "transaction",
                    3: "termination"}.get(rank, "standalone")
        result[fid] = {
            "effectiveDate": r.get("effectiveDate"),
            "documentType": r.get("documentType", "agreement"),
            "documentRank": r.get("documentRank", 1),
            "functionalRole": role,
            "functionalRoleRank": _ROLE_RANK.get(role, 1),
            "riskFlags": r.get("riskFlags", []),
        }
    return result


def _inject_unindexed_files(conversation_id: str, file_meta: Dict) -> None:
    """
    Find files in filePages that have pageWiseText (Step 2 of the worker is done)
    but whose fileId does not appear in fileSections at all (Step 3/4 not yet done,
    or embedding failed silently).

    For each such file, synthesise one fileSections document per page from
    pageWiseText and write them into MongoDB.  This makes the content available
    to Phase 2 injectors (filename-hint, financial value, WO structural) which
    query MongoDB directly — even before the Qdrant embedding completes.

    Also updates file_meta so Phase 4 sorts these synthetic sections correctly.

    This is idempotent: once real sections arrive (from the worker) they will
    exist alongside the synthetics; the worker will then set sections_embedded=True
    and the synthetics will never be re-created.
    """
    import json as _json
    from src.utils.connection_utils import db

    # Files registered in filePages for this conversation (with or without pageWiseText)
    all_fp = list(db["filePages"].find(
        {"conversationId": conversation_id},
        {"fileId": 1, "fileName": 1, "pageWiseText": 1,
         "functionalRole": 1, "documentRank": 1, "effectiveDate": 1,
         "sections_embedded": 1},
    ))
    if not all_fp:
        return

    # Which files already have at least one section in fileSections (real or synthetic)?
    indexed_file_ids = {
        r["fileId"]
        for r in db["fileSections"].find(
            {"conversationId": conversation_id},
            {"fileId": 1},
        )
    }

    # Also treat files flagged sections_embedded=True as indexed — their sections
    # existed at some point.  The only case we must synthesise is when fileSections
    # has NO rows for the fileId AND pageWiseText is available.
    # This handles the wipe scenario: sections_embedded=True but fileSections empty.

    _ROLE_RANK = {
        "master_agreement": 0, "standalone": 0,
        "modification": 1, "transaction": 2, "termination": 3,
    }

    new_docs = []
    for fp in all_fp:
        fid = fp.get("fileId")
        fname = fp.get("fileName", "")
        if not fid:
            continue

        # Skip if already in fileSections
        if fid in indexed_file_ids:
            continue

        raw = fp.get("pageWiseText", "")
        if not raw:
            # pageWiseText absent from local DB (file was processed against old Atlas instance).
            # Try downloading from blob storage and extracting text on-the-fly so the
            # file is available for this query without requiring a re-upload.
            raw = _fetch_page_wise_text_from_blob(fid, fname, fp.get("blobName", ""), conversation_id)
            if not raw:
                continue
        try:
            pages: dict = _json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not pages:
            continue

        role = fp.get("functionalRole") or ""
        if role not in _ROLE_RANK:
            rank = fp.get("documentRank", 1)
            role = {0: "master_agreement", 1: "modification", 2: "transaction",
                    3: "termination"}.get(rank, "standalone")

        # Update file_meta so Phase 4 sorts these files correctly
        if fid not in file_meta:
            file_meta[fid] = {
                "effectiveDate": fp.get("effectiveDate"),
                "documentType": "agreement",
                "documentRank": fp.get("documentRank", 1),
                "functionalRole": role,
                "functionalRoleRank": _ROLE_RANK.get(role, 1),
                "riskFlags": [],
            }

        for page_str, page_text in pages.items():
            if not page_text or not page_text.strip():
                continue
            try:
                page_num = int(page_str)
            except (ValueError, TypeError):
                page_num = 0
            sid = f"__raw_p{page_num}"
            new_docs.append({
                "conversationId": conversation_id,
                "fileId": fid,
                "fileName": fname,
                "sectionId": sid,
                "sectionTitle": f"Page {page_num}",
                "content": page_text.strip(),
                "pageNumber": page_num,
                "clauseType": "other",
                "scheduleContext": "",
                "riskLevel": 1,
                "riskNote": "",
                "_synthetic": True,  # marks these as pre-embedding fallback docs
            })

    if new_docs:
        # Upsert by (conversationId, fileId, sectionId) to avoid duplicates
        from pymongo import UpdateOne
        ops = [
            UpdateOne(
                {"conversationId": d["conversationId"],
                 "fileId": d["fileId"],
                 "sectionId": d["sectionId"]},
                {"$setOnInsert": d},
                upsert=True,
            )
            for d in new_docs
        ]
        try:
            db["fileSections"].bulk_write(ops, ordered=False)
            unique_files = {d["fileId"] for d in new_docs}
            logger.info(
                "[SectionRetriever] Pre-phase: synthesised %d raw-page section(s) "
                "for %d unindexed file(s): %s",
                len(new_docs), len(unique_files),
                [d["fileName"] for d in new_docs
                 if d["fileId"] in unique_files and d["sectionId"] == "__raw_p1"][:5],
            )
        except Exception as e:
            logger.warning("[SectionRetriever] Pre-phase bulk_write failed: %s", e)


def retrieve_sections(
    conversation_id: str,
    query: str,
    top_k: int = 20,
) -> Tuple[str, Dict]:
    """
    Multi-hop section retrieval. Returns (context_text, citation_metadata).
    Returns ("", {}) on failure so the caller can fall back gracefully.
    """
    try:
        # Load per-file temporal metadata once — used in Phase 4 for version-aware ordering
        file_meta = _build_file_meta_map(conversation_id)

        # Pre-phase: detect files that are registered in filePages but whose sections
        # have not yet been indexed in fileSections (still processing or embedding failed).
        # For those files, synthesise synthetic sections from pageWiseText so their
        # content is ALWAYS available regardless of worker timing.
        _inject_unindexed_files(conversation_id, file_meta)

        # Pre-phase: Entity alias resolution via graph
        # Expands "Mainstay HK" → ["Mainstay HK", "Mainstay Asia Ltd", "Mainstay (HK) Limited"]
        # so all aliases are searched simultaneously rather than relying on similarity.
        alias_augmented_query = query
        try:
            from src.services.graph_extractor import resolve_entity_aliases
            aliases = resolve_entity_aliases(conversation_id, query)
            if aliases:
                alias_str = " OR ".join(f'"{a}"' for a in aliases[:4])
                alias_augmented_query = f"{query} [{alias_str}]"
                logger.info(
                    "[SectionRetriever] Entity aliases resolved: %s", aliases[:4]
                )
        except Exception as alias_e:
            logger.debug("[SectionRetriever] Alias resolution skipped: %s", alias_e)

        # Phase 0: Expand query into sub-queries for broader semantic coverage
        expanded_queries = _phase0_query_expansion(alias_augmented_query)

        # Phase 1: Semantic entry via Qdrant (one search per sub-query, merged)
        seed_sections = _phase1_multi_semantic(conversation_id, expanded_queries, top_k, file_meta)
        if not seed_sections:
            logger.warning(
                f"[SectionRetriever] Phase 1 returned no sections for conv={conversation_id}"
            )
            return "", {}

        # Phase 2: Cross-ref expansion (1 hop) + all injections
        all_candidates = _phase2_crossref(conversation_id, seed_sections, file_meta, query)

        # Phase 2 (graph): Inject sections linked via semantic edges
        # (CONDITIONS, SUPERSEDES, EXCEPTIONS, DEFINES) from the knowledge graph.
        # These are sections structurally connected to the seed sections that
        # semantic search would miss — e.g. Clause 6.5 is conditioned by 6.1.
        try:
            import hashlib as _hl
            def _gnid(t, *p):
                raw = f"{t}:" + ":".join(p)
                return _hl.md5(raw.encode()).hexdigest()[:16]

            from src.services.graph_extractor import get_graph_context_for_sections
            seed_node_ids = [
                _gnid("section", s.get("fileId", ""), s.get("sectionId", ""))
                for s in seed_sections
                if s.get("fileId") and s.get("sectionId")
            ]
            graph_injected = get_graph_context_for_sections(
                conversation_id, seed_node_ids
            )
            if graph_injected:
                existing_keys = {
                    (s.get("fileId"), s.get("sectionId")) for s in all_candidates
                }
                for sec in graph_injected:
                    key = (sec.get("fileId"), sec.get("sectionId"))
                    if key not in existing_keys:
                        sec["_graph_injection"] = True
                        sec["score"] = 0.8  # treat as high-confidence injection
                        all_candidates.append(sec)
                        existing_keys.add(key)
                logger.info(
                    "[SectionRetriever] Graph injection: +%d section(s) via semantic edges",
                    len(graph_injected),
                )
        except Exception as graph_e:
            logger.debug("[SectionRetriever] Graph injection skipped: %s", graph_e)

        # Phase 3: LLM navigation — prune to most relevant
        confirmed = _phase3_llm_navigate(query, all_candidates)
        if not confirmed:
            confirmed = seed_sections  # safe fallback

        # Phase 3.5: Gap analysis — LLM identifies missing sections after reviewing evidence
        gap_sections = _phase35_gap_analysis(query, confirmed, conversation_id)
        if gap_sections:
            confirmed = confirmed + gap_sections

        # Phase 4: Assemble context with citations
        context_text, citation_metadata = _phase4_assemble(confirmed, file_meta)

        # Phase 4b: Normalize any fee/SLA tables for cleaner LLM reading
        context_text = _phase4b_normalize_tables(context_text)

        return context_text, citation_metadata

    except Exception as e:
        logger.error(f"[SectionRetriever] retrieve_sections failed: {e}")
        return "", {}


# ---------------------------------------------------------------------------
# Phase 1: Semantic search
# ---------------------------------------------------------------------------

def _phase1_multi_semantic(
    conversation_id: str,
    queries: List[str],
    top_k: int,
    file_meta: Dict,
) -> List[dict]:
    """
    Run _phase1_semantic for each expanded query and merge results, keeping
    the highest semantic score per unique (fileId, sectionId) pair.
    """
    if len(queries) == 1:
        return _phase1_semantic(conversation_id, queries[0], top_k, file_meta)

    best: Dict[tuple, dict] = {}
    for q in queries:
        for sec in _phase1_semantic(conversation_id, q, top_k, file_meta):
            key = (sec["fileId"], sec["sectionId"])
            if key not in best or sec["score"] > best[key]["score"]:
                best[key] = sec

    merged = list(best.values())

    # Drop low-confidence tail — sections scoring below 0.35 across all sub-queries
    # are semantically marginal and bloat the prompt without adding value.
    # The Phase 2 injection pipeline (cross-refs, clauseType siblings, order_of_precedence)
    # will surface any truly relevant sections that fall below this threshold.
    _MIN_SCORE = 0.35
    before = len(merged)
    merged = [s for s in merged if s["score"] >= _MIN_SCORE]
    if len(merged) < before:
        logger.info(
            "[SectionRetriever] Phase 1 multi: score threshold %.2f dropped %d low-confidence section(s)",
            _MIN_SCORE, before - len(merged),
        )

    logger.info(
        "[SectionRetriever] Phase 1 multi: %d sub-queries → %d unique section(s)",
        len(queries), len(merged),
    )
    return merged


def _phase1_semantic(conversation_id: str, query: str, top_k: int, file_meta: Dict) -> List[dict]:
    """
    Embed the query and search contract_sections filtered to this conversation.
    Full section content is loaded from MongoDB (Qdrant stores metadata only).
    """
    import ssl
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
    from src.utils.sentence_transformer_instance import get_sentence_transformer
    from src.utils.qdrant.qdrant_utils import get_qdrant_client
    from src.utils.connection_utils import db
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    def _is_transient_network_error(exc: BaseException) -> bool:
        """Retry on SSL errors and common transient network exceptions."""
        if isinstance(exc, ssl.SSLError):
            return True
        msg = str(exc).lower()
        return any(kw in msg for kw in ("sslv3", "bad record mac", "connection reset", "broken pipe", "timed out", "temporarily unavailable"))

    model = get_sentence_transformer()
    query_vec = model.encode(query).tolist()

    client = get_qdrant_client()

    @retry(
        retry=retry_if_exception(_is_transient_network_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def _query_with_retry():
        return client.query_points(
            collection_name=_SECTION_COLLECTION,
            query=query_vec,
            query_filter=Filter(must=[
                FieldCondition(
                    key="conversationId",
                    match=MatchValue(value=conversation_id),
                )
            ]),
            limit=top_k,
            with_payload=True,
        )

    try:
        response = _query_with_retry()
    except Exception as e:
        logger.error(f"[SectionRetriever] Qdrant query failed after retries: {e}")
        return []

    results = response.points

    if not results:
        return []

    # Batch-fetch full content from MongoDB
    section_ids = [r.payload["sectionId"] for r in results]
    docs = list(db["fileSections"].find({
        "conversationId": conversation_id,
        "sectionId": {"$in": section_ids},
    }))
    # Build lookup keyed by (fileId, sectionId) to handle same sectionId in different files
    doc_map = {(d["fileId"], d["sectionId"]): d for d in docs}

    sections = []
    for r in results:
        p = r.payload
        doc = doc_map.get((p["fileId"], p["sectionId"]))
        if doc:
            fmeta = file_meta.get(doc["fileId"], {})
            # Prefer file_meta (from filePages) for freshness; fall back to
            # Qdrant payload values written at embed time (Fix 1).
            sections.append({
                "sectionId": doc["sectionId"],
                "sectionTitle": doc.get("sectionTitle", ""),
                "fileId": doc["fileId"],
                "fileName": doc.get("fileName", ""),
                "pageNumber": doc.get("pageNumber", 0),
                "content": doc.get("content", ""),
                "clauseType": doc.get("clauseType", "other"),
                "scheduleContext": doc.get("scheduleContext", ""),
                "riskLevel": doc.get("riskLevel", 1),
                "riskNote": doc.get("riskNote", ""),
                "effectiveDate": fmeta.get("effectiveDate") or p.get("effectiveDate"),
                "documentType": fmeta.get("documentType", "agreement"),
                "documentRank": fmeta.get("documentRank") if fmeta else p.get("documentRank", 1),
                "functionalRole": fmeta.get("functionalRole") or p.get("functionalRole", "standalone"),
                "functionalRoleRank": fmeta.get("functionalRoleRank") if fmeta else p.get("functionalRoleRank", 1),
                "score": r.score,
            })

    logger.info(
        f"[SectionRetriever] Phase 1: {len(sections)} section(s) found "
        f"for conv={conversation_id}"
    )
    return sections


# ---------------------------------------------------------------------------
# Phase 2: Cross-reference expansion
# ---------------------------------------------------------------------------

def _phase2_crossref(
    conversation_id: str,
    seed_sections: List[dict],
    file_meta: Dict,
    query: str = "",
) -> List[dict]:
    """
    Follow one level of resolved cross-refs from fileCrossRefs and add the
    target sections to the candidate pool.

    Also always includes sections classified as order_of_precedence — these
    define which document wins in a conflict and are rarely surfaced by
    semantic search alone (their titles are often structural rather than
    descriptive of their legal significance).

    Additional injections driven by query content:
      - Named schedule injection: if the query mentions a schedule/appendix by
        name or number, all sections of that schedule are injected. Solves the
        "what does Schedule 4 cover?" class of queries where Phase 1 may return
        only the schedule header without its content sub-sections.
      - Financial value injection: if the query asks about fees, charges, rates,
        or percentages, sections containing currency amounts or percentage figures
        are injected directly from MongoDB. Solves the "£90,000 not found" class
        of failures where a fee table scores low in semantic search.
    """
    from src.services.crossref_extractor import (
        get_cross_refs_for_sections,
        fetch_resolved_sections,
    )
    from src.utils.connection_utils import db

    if not seed_sections:
        return seed_sections

    seen = {(s["fileId"], s["sectionId"]) for s in seed_sections}
    expanded = list(seed_sections)

    # Always pull in order_of_precedence sections — they answer "which document
    # wins in a conflict?" and are systematically missed by semantic search.
    precedence_docs = list(db["fileSections"].find({
        "conversationId": conversation_id,
        "clauseType": "order_of_precedence",
    }))
    for sec in precedence_docs:
        key = (sec["fileId"], sec["sectionId"])
        if key not in seen:
            seen.add(key)
            expanded.append({
                "sectionId": sec["sectionId"],
                "sectionTitle": sec.get("sectionTitle", ""),
                "fileId": sec["fileId"],
                "fileName": sec.get("fileName", ""),
                "pageNumber": sec.get("pageNumber", 0),
                "content": sec.get("content", ""),
                "score": 0.0,
            })
    if precedence_docs:
        logger.info(
            f"[SectionRetriever] Phase 2: injected {len(precedence_docs)} "
            f"order_of_precedence section(s)"
        )

    # Always inject governing_law sections — these are systematically missed because
    # "governing law" / "choice of law" sections use formulaic language ("this Agreement
    # shall be governed by the laws of...") that scores low against casual queries like
    # "what is the governing law?". Same rationale as order_of_precedence injection.
    governing_law_docs = list(db["fileSections"].find({
        "conversationId": conversation_id,
        "clauseType": "governing_law",
    }))
    for sec in governing_law_docs:
        key = (sec["fileId"], sec["sectionId"])
        if key not in seen:
            seen.add(key)
            fmeta = file_meta.get(sec["fileId"], {})
            expanded.append({
                "sectionId": sec["sectionId"],
                "sectionTitle": sec.get("sectionTitle", ""),
                "fileId": sec["fileId"],
                "fileName": sec.get("fileName", ""),
                "pageNumber": sec.get("pageNumber", 0),
                "content": sec.get("content", ""),
                "clauseType": "governing_law",
                "score": 0.0,
                "_governing_law_injection": True,  # immune to Phase 3 pruning
            })
    if governing_law_docs:
        logger.info(
            f"[SectionRetriever] Phase 2: injected {len(governing_law_docs)} "
            f"governing_law section(s)"
        )

    # Always inject dispute_resolution sections — §35-style escalation/dispute
    # clauses use formal procedural language that scores low against natural
    # queries like "what happens if there is a dispute?". Same rationale as
    # governing_law injection.
    # Dual query: match on clauseType OR on title keywords — the latter catches
    # sections the BERT classifier mislabeled (e.g. as "general") because 217
    # training examples is thin for atypical headings like "Escalation Procedure".
    _DISPUTE_TITLE_RE = re.compile(
        r'\b(dispute|escalat|arbitrat|mediat|litigation|expert\s+determination'
        r'|adjudicat|resolution\s+procedure|complaints?\s+procedure)\b',
        re.IGNORECASE,
    )
    dispute_docs = list(db["fileSections"].find({
        "conversationId": conversation_id,
        "$or": [
            {"clauseType": "dispute_resolution"},
            {"sectionTitle": {"$regex": r"(?i)(dispute|escalat|arbitrat|mediat|litigation|expert\s+determination)"}},
        ],
    }))
    for sec in dispute_docs:
        key = (sec["fileId"], sec["sectionId"])
        if key not in seen:
            seen.add(key)
            expanded.append({
                "sectionId": sec["sectionId"],
                "sectionTitle": sec.get("sectionTitle", ""),
                "fileId": sec["fileId"],
                "fileName": sec.get("fileName", ""),
                "pageNumber": sec.get("pageNumber", 0),
                "content": sec.get("content", ""),
                "clauseType": "dispute_resolution",
                "score": 0.0,
                "_dispute_resolution_injection": True,  # immune to Phase 3 pruning
            })
    if dispute_docs:
        logger.info(
            f"[SectionRetriever] Phase 2: injected {len(dispute_docs)} "
            f"dispute_resolution section(s) (clauseType + title-keyword match)"
        )

    # SLA-aware injector — when the query is about service levels, uptime, or
    # service credits, inject warranties and payment_terms sections.  SLA credit
    # schedules (e.g. Schedule 1 §5.1) are tagged payment_terms/warranties but
    # score poorly against semantic queries because they are presented as tables
    # rather than prose.
    _SLA_QUERY_RE = re.compile(
        r'\b(sla|service[\s\-]?level|uptime|availability|service[\s\-]?credit'
        r'|credit[s]?\s+(?:for|due)|remedies?\s+for\s+(?:downtime|outage)'
        r'|performance\s+(?:credit|penalty|standard))\b',
        re.IGNORECASE,
    )
    if query and _SLA_QUERY_RE.search(query):
        sla_docs = list(db["fileSections"].find({
            "conversationId": conversation_id,
            "clauseType": {"$in": ["warranties", "payment_terms", "service_levels"]},
        }))
        added_sla = 0
        for sec in sla_docs:
            key = (sec["fileId"], sec["sectionId"])
            if key not in seen:
                seen.add(key)
                expanded.append({
                    "sectionId": sec["sectionId"],
                    "sectionTitle": sec.get("sectionTitle", ""),
                    "fileId": sec["fileId"],
                    "fileName": sec.get("fileName", ""),
                    "pageNumber": sec.get("pageNumber", 0),
                    "content": sec.get("content", ""),
                    "clauseType": sec.get("clauseType", ""),
                    "score": 0.0,
                    "_sla_injection": True,  # immune to Phase 3 pruning
                })
                added_sla += 1
        if added_sla:
            logger.info(
                f"[SectionRetriever] Phase 2: SLA query detected — injected "
                f"{added_sla} warranties/payment_terms/service_levels section(s)"
            )

    # Preamble injection for party/date queries.
    # The document preamble (introductory paragraph with parties, effective date,
    # recitals) often has no section heading and therefore no Qdrant vector.
    # When the query asks about parties, dates, or deal identities, inject:
    #   1. Any section with sectionId == "PREAMBLE" from fileSections (post-fix uploads)
    #   2. Raw page 1 text from filePages.pageWiseText as a fallback for legacy uploads
    _PARTY_QUERY_RE = re.compile(
        r'\b(party|parties|buyer|seller|purchaser|vendor|grantor|grantee'
        r'|effective\s+date|date\s+of\s+(?:this\s+)?agreement|entered\s+into'
        r'|made\s+(?:as\s+of|on)|between\s+and|share|shares|units|recital)\b',
        re.IGNORECASE,
    )
    if query and _PARTY_QUERY_RE.search(query):
        # 1. Inject PREAMBLE sections from fileSections
        preamble_sections = list(db["fileSections"].find({
            "conversationId": conversation_id,
            "sectionId": "PREAMBLE",
        }))
        added_preamble = 0
        for sec in preamble_sections:
            key = (sec["fileId"], sec["sectionId"])
            if key not in seen:
                seen.add(key)
                fmeta = file_meta.get(sec["fileId"], {})
                expanded.append({
                    "sectionId": sec["sectionId"],
                    "sectionTitle": sec.get("sectionTitle", "Preamble / Parties"),
                    "fileId": sec["fileId"],
                    "fileName": sec.get("fileName", ""),
                    "pageNumber": sec.get("pageNumber", 1),
                    "content": sec.get("content", ""),
                    "clauseType": sec.get("clauseType", "general"),
                    "score": 0.0,
                    "_preamble_injection": True,
                })
                added_preamble += 1
        # 2. Fallback: inject raw preamble page from pageWiseText for legacy files
        #    that were parsed before the PREAMBLE fix.
        #    We scan pages in order and pick the first one that:
        #      (a) contains preamble signature language ("This Agreement", "entered into as of"…)
        #      (b) is NOT a TOC page (few dotted leader lines)
        #    This handles contracts with a cover page or TOC before the actual agreement text.
        import json as _json

        _TOC_LINE_RE_R = re.compile(r'\.{4,}\s*\d+\s*$')
        _PREAMBLE_SIG_RE_R = re.compile(
            r'\b(this\s+agreement|entered\s+into|made\s+(?:as\s+of|on)|'
            r'hereby\s+agree|witnesseth|whereas|by\s+and\s+between|'
            r'share\s+purchase|purchase\s+agreement|employment\s+agreement|'
            r'non[\-\s]disclosure)\b',
            re.IGNORECASE,
        )

        def _is_toc_page_r(text: str) -> bool:
            lines = [l for l in text.split('\n') if l.strip()]
            if not lines:
                return False
            hits = sum(1 for l in lines if _TOC_LINE_RE_R.search(l))
            return hits >= 3 and hits / len(lines) >= 0.25

        def _find_preamble_page(pages: dict) -> tuple:
            """Return (page_num_int, page_text) for the first preamble page."""
            def _pk(k):
                return int(re.sub(r'[^0-9]', '', k) or '0')
            for k in sorted(pages.keys(), key=_pk):
                text = pages[k] or ""
                if _PREAMBLE_SIG_RE_R.search(text) and not _is_toc_page_r(text):
                    return _pk(k), text
            # fallback: first non-empty page
            for k in sorted(pages.keys(), key=_pk):
                if pages[k] and pages[k].strip():
                    return _pk(k), pages[k]
            return 1, ""

        if not preamble_sections:
            fp_records = list(db["filePages"].find(
                {"conversationId": conversation_id},
                {"fileId": 1, "fileName": 1, "pageWiseText": 1},
            ))
            for fp in fp_records:
                fid = fp.get("fileId")
                fname = fp.get("fileName", "")
                raw = fp.get("pageWiseText", "")
                if not raw or not fid:
                    continue
                try:
                    pages: dict = _json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    continue
                page_num_found, page1_text = _find_preamble_page(pages)
                if not page1_text or not page1_text.strip():
                    continue
                synthetic_id = "__raw_page1_preamble"
                key = (fid, synthetic_id)
                if key in seen:
                    continue
                seen.add(key)
                fmeta = file_meta.get(fid, {})
                expanded.append({
                    "sectionId": synthetic_id,
                    "sectionTitle": "Page 1 (Preamble / Parties)",
                    "fileId": fid,
                    "fileName": fname,
                    "pageNumber": page_num_found,
                    "content": page1_text.strip()[:3000],  # cap to avoid flooding
                    "clauseType": "general",
                    "score": 0.0,
                    "_preamble_injection": True,
                })
                added_preamble += 1
        if added_preamble:
            logger.info(
                f"[SectionRetriever] Phase 2: preamble injection added {added_preamble} "
                f"section(s) for party/date query"
            )

    # Inject qualifier sections from files already in the seed set.
    # These are sections flagged at index time as containing hedging/limitation
    # language (e.g. "makes no warranty", "notwithstanding", "exclusive remedy").
    # Scoped to seed files only so we don't flood context with unrelated caveats.
    # Inject qualifier sections (hedging/limitation language) from seed files.
    # Capped at 5 to avoid flooding Phase 3 with every "notwithstanding" clause
    # across all three documents — targeted, not exhaustive.
    _MAX_QUALIFIER_INJECTIONS = 5
    seed_file_ids = list({s["fileId"] for s in seed_sections})
    qualifier_docs = list(db["fileSections"].find({
        "conversationId": conversation_id,
        "fileId": {"$in": seed_file_ids},
        "qualifierFlag": True,
    }))
    added_qualifiers = 0
    for sec in qualifier_docs:
        if added_qualifiers >= _MAX_QUALIFIER_INJECTIONS:
            break
        key = (sec["fileId"], sec["sectionId"])
        if key not in seen:
            seen.add(key)
            expanded.append({
                "sectionId": sec["sectionId"],
                "sectionTitle": sec.get("sectionTitle", ""),
                "fileId": sec["fileId"],
                "fileName": sec.get("fileName", ""),
                "pageNumber": sec.get("pageNumber", 0),
                "content": sec.get("content", ""),
                "scheduleContext": sec.get("scheduleContext", ""),
                "score": 0.0,
            })
            added_qualifiers += 1
    if added_qualifiers:
        logger.info(
            f"[SectionRetriever] Phase 2: injected {added_qualifiers} "
            f"qualifier/limitation section(s)"
        )

    seed_ids = [s["sectionId"] for s in seed_sections]

    # Query cross-refs once per unique source file
    for source_file_id in {s["fileId"] for s in seed_sections}:
        refs = get_cross_refs_for_sections(
            conversation_id, seed_ids, source_file_id=source_file_id
        )
        for ref in refs:
            resolved_fid = ref.get("resolvedFileId")
            resolved_sid = ref.get("resolvedSectionId")
            if not resolved_fid or not resolved_sid:
                continue
            key = (resolved_fid, resolved_sid)
            if key in seen:
                continue
            target_docs = fetch_resolved_sections(
                conversation_id, [resolved_sid], file_id=resolved_fid
            )
            for sec in target_docs:
                k = (sec["fileId"], sec["sectionId"])
                if k not in seen:
                    seen.add(k)
                    expanded.append({
                        "sectionId": sec["sectionId"],
                        "sectionTitle": sec.get("sectionTitle", ""),
                        "fileId": sec["fileId"],
                        "fileName": sec.get("fileName", ""),
                        "pageNumber": sec.get("pageNumber", 0),
                        "content": sec.get("content", ""),
                        "score": 0.0,  # cross-ref expansion, no semantic score
                    })

    if len(expanded) > len(seed_sections):
        logger.info(
            f"[SectionRetriever] Phase 2: {len(seed_sections)} seed → "
            f"{len(expanded)} after cross-ref expansion"
        )

    # Reverse cross-ref lookup — find sections that cite our seed sections.
    # Catches "notwithstanding clause X", "subject to clause X", and other
    # override/exception clauses that reference the seed but are never
    # reachable via forward expansion alone.
    reverse_hits = list(db["fileCrossRefs"].find({
        "conversationId": conversation_id,
        "resolvedFileId": {"$in": [s["fileId"] for s in seed_sections]},
        "resolvedSectionId": {"$in": [s["sectionId"] for s in seed_sections]},
        "sourceFileId": {"$ne": None},
        "sourceSectionId": {"$ne": None},
    }))

    if reverse_hits:
        reverse_section_ids = list({r["sourceSectionId"] for r in reverse_hits})
        reverse_file_ids = list({r["sourceFileId"] for r in reverse_hits})
        reverse_docs = list(db["fileSections"].find({
            "conversationId": conversation_id,
            "fileId": {"$in": reverse_file_ids},
            "sectionId": {"$in": reverse_section_ids},
        }))
        added_reverse = 0
        for sec in reverse_docs:
            key = (sec["fileId"], sec["sectionId"])
            if key not in seen:
                seen.add(key)
                expanded.append({
                    "sectionId": sec["sectionId"],
                    "sectionTitle": sec.get("sectionTitle", ""),
                    "fileId": sec["fileId"],
                    "fileName": sec.get("fileName", ""),
                    "pageNumber": sec.get("pageNumber", 0),
                    "content": sec.get("content", ""),
                    "score": 0.0,
                })
                added_reverse += 1
        if added_reverse:
            logger.info(
                f"[SectionRetriever] Phase 2: reverse cross-ref added {added_reverse} "
                f"override/notwithstanding section(s)"
            )

    # --- Named schedule injection ---
    # If the query explicitly references a schedule or appendix by name/number,
    # pull ALL sections of that schedule from MongoDB regardless of semantic score.
    # Tagged with scheduleContext so Phase 3 keeps them even if headings are opaque.
    if query:
        _SCHED_REF_RE = re.compile(
            r'\b(schedule|appendix|exhibit|annex)\s+(\d+|[A-Z])\b',
            re.IGNORECASE,
        )
        schedule_refs = _SCHED_REF_RE.findall(query.lower())
        for sched_type, sched_num in schedule_refs:
            sched_label = f"{sched_type.capitalize()} {sched_num.upper()}"
            sched_sections = list(db["fileSections"].find({
                "conversationId": conversation_id,
                "scheduleContext": re.compile(
                    rf'\b{re.escape(sched_type)}\s+{re.escape(sched_num)}\b',
                    re.IGNORECASE,
                ),
            }, limit=30))
            added_sched = 0
            for sec in sched_sections:
                key = (sec["fileId"], sec["sectionId"])
                if key not in seen:
                    seen.add(key)
                    fmeta = file_meta.get(sec["fileId"], {})
                    expanded.append({
                        "sectionId": sec["sectionId"],
                        "sectionTitle": sec.get("sectionTitle", ""),
                        "fileId": sec["fileId"],
                        "fileName": sec.get("fileName", ""),
                        "pageNumber": sec.get("pageNumber", 0),
                        "content": sec.get("content", ""),
                        "clauseType": sec.get("clauseType", "other"),
                        "scheduleContext": sec.get("scheduleContext", sched_label),
                        "riskLevel": sec.get("riskLevel", 1),
                        "riskNote": sec.get("riskNote", ""),
                        "effectiveDate": fmeta.get("effectiveDate"),
                        "documentType": fmeta.get("documentType", ""),
                        "documentRank": fmeta.get("documentRank", 1),
                        "functionalRole": fmeta.get("functionalRole", "standalone"),
                        "functionalRoleRank": fmeta.get("functionalRoleRank", 1),
                        "score": 0.0,
                    })
                    added_sched += 1
            if added_sched:
                logger.info(
                    f"[SectionRetriever] Phase 2: named schedule injection '{sched_label}' "
                    f"added {added_sched} section(s)"
                )

    # --- Work Order structural injection ---
    # When ANY Work Order file appears in the candidate set (functionalRole=transaction),
    # force-inject its §6 (Service Fee) and §10 (Deviations from Agreement) sections.
    # These are the two clauses most commonly asked about in Work Order queries but most
    # commonly missed by Phase 1 semantic search because:
    #   §6 — fee tables have weak embeddings ("HK$163,914" ≠ "service fee")
    #   §10 — deviations section is absent in older templates, present in newer ones,
    #          so its absence is as significant as its presence
    # Scoped to candidate files only (not the full corpus) to avoid flooding context.
    candidate_file_ids = list({s["fileId"] for s in expanded})
    wo_files = list(db["filePages"].find(
        {
            "conversationId": conversation_id,
            "fileId": {"$in": candidate_file_ids},
            "functionalRole": "transaction",
        },
        {"fileId": 1},
    ))
    wo_file_ids = [r["fileId"] for r in wo_files if r.get("fileId")]
    if wo_file_ids:
        # Match sectionId patterns: "6", "6.x", "10", "10.x"
        structural_sections = list(db["fileSections"].find({
            "conversationId": conversation_id,
            "fileId": {"$in": wo_file_ids},
            "sectionId": re.compile(r'^(6|6\.\d+|10|10\.\d+)$'),
        }))
        added_structural = 0
        for sec in structural_sections:
            key = (sec["fileId"], sec["sectionId"])
            if key in seen:
                continue
            seen.add(key)
            fmeta = file_meta.get(sec["fileId"], {})
            expanded.append({
                "sectionId": sec["sectionId"],
                "sectionTitle": sec.get("sectionTitle", ""),
                "fileId": sec["fileId"],
                "fileName": sec.get("fileName", ""),
                "pageNumber": sec.get("pageNumber", 0),
                "content": sec.get("content", ""),
                "clauseType": sec.get("clauseType", "other"),
                "scheduleContext": sec.get("scheduleContext", ""),
                "riskLevel": sec.get("riskLevel", 1),
                "riskNote": sec.get("riskNote", ""),
                "effectiveDate": fmeta.get("effectiveDate"),
                "documentType": fmeta.get("documentType", ""),
                "documentRank": fmeta.get("documentRank", 1),
                "functionalRole": fmeta.get("functionalRole", "transaction"),
                "functionalRoleRank": fmeta.get("functionalRoleRank", 2),
                "score": 0.0,
                "_wo_structural": True,  # immune to Phase 3 pruning
            })
            added_structural += 1
        if added_structural:
            logger.info(
                "[SectionRetriever] Phase 2: WO structural injection added %d "
                "§6/§10 section(s) from %d transaction file(s)",
                added_structural, len(wo_file_ids),
            )

    # --- Financial value injection ---
    # When query asks about fees, charges, rates, prices, or percentages, pull sections
    # containing currency amounts or percentage figures from MongoDB.  These score low
    # in semantic search because table cells like "£90,000" have weak embeddings, but
    # they are the direct answer to fee/rate questions.
    _FINANCIAL_QUERY_RE = re.compile(
        r'\b(fee|charge|rate|cost|price|amount|salary|pay|remunerat|licen[sc]e|hosting'
        r'|interest|uptime|availab|sla|service\s+level|percentage|credit)\b',
        re.IGNORECASE,
    )
    _CURRENCY_RE = re.compile(
        r'(?:£|€|\$|USD|GBP|EUR|HKD|SGD|AUD)\s*[\d,]+|[\d,]+\s*(?:£|€|\$|USD|GBP|EUR)',
        re.IGNORECASE,
    )
    _PERCENTAGE_RE = re.compile(r'\d+(?:\.\d+)?\s*%')

    if query and _FINANCIAL_QUERY_RE.search(query):
        # Search ALL files in the conversation for sections containing financial values.
        # The previous behaviour (single highest-rank file only) caused fees defined in
        # the master agreement (e.g. Clause 6.5 conversion fee table) to be missed when a
        # Work Order was the highest-rank seed file.
        # The LLM's specificity hierarchy (Schedule > body; transaction > master) already
        # handles conflicting rates from different document levels, so we can safely include
        # financial sections from the full corpus and let the LLM resolve precedence.
        fin_sections = list(db["fileSections"].find(
            {"conversationId": conversation_id}, limit=300
        ))
        # Sort by descending functionalRoleRank so higher-rank docs fill the budget first,
        # preserving the original preference for transaction-level rates when there is a
        # genuine conflict — but without excluding master-agreement fee schedules entirely.
        fin_sections.sort(
            key=lambda s: file_meta.get(s["fileId"], {}).get("functionalRoleRank", 1),
            reverse=True,
        )
        added_fin = 0
        _MAX_FIN_INJECTIONS = 15
        for sec in fin_sections:
            if added_fin >= _MAX_FIN_INJECTIONS:
                break
            content = sec.get("content", "")
            if not (_CURRENCY_RE.search(content) or _PERCENTAGE_RE.search(content)):
                continue
            key = (sec["fileId"], sec["sectionId"])
            if key in seen:
                continue
            seen.add(key)
            fmeta = file_meta.get(sec["fileId"], {})
            expanded.append({
                "sectionId": sec["sectionId"],
                "sectionTitle": sec.get("sectionTitle", ""),
                "fileId": sec["fileId"],
                "fileName": sec.get("fileName", ""),
                "pageNumber": sec.get("pageNumber", 0),
                "content": content,
                "clauseType": sec.get("clauseType", "other"),
                "scheduleContext": sec.get("scheduleContext", ""),
                "riskLevel": sec.get("riskLevel", 1),
                "riskNote": sec.get("riskNote", ""),
                "effectiveDate": fmeta.get("effectiveDate"),
                "documentType": fmeta.get("documentType", ""),
                "documentRank": fmeta.get("documentRank", 1),
                "functionalRole": fmeta.get("functionalRole", "standalone"),
                "functionalRoleRank": fmeta.get("functionalRoleRank", 1),
                "score": 0.0,
                "_financial_injection": True,  # immune to Phase 3 LLM pruning
            })
            added_fin += 1

        if added_fin:
            logger.info(
                f"[SectionRetriever] Phase 2: financial value injection added "
                f"{added_fin} section(s) across all files in conversation"
            )

    # --- Filename-hint injection ---
    # If any token sequence in the query matches part of a document filename in the
    # conversation, inject ALL sections of that document.  Solves the class of query
    # "was the notice period followed for Joi Clark?" where the termination letter's
    # filename contains "Joi Clark" but its sections score low in Phase 1 because
    # the query is primarily about notice periods and the letter is short/informal.
    #
    # Uses fuzzy token matching (difflib.SequenceMatcher) so that name spelling
    # variants like "MacDonald" vs "McDonald" are treated as the same person.
    # A token pair is considered a match if similarity >= 0.82 OR if one contains
    # the other (handles "Alex" matching "Alexander").
    if query:
        from difflib import SequenceMatcher

        def _fuzzy_token_match(fname_token: str, query_lower: str) -> bool:
            """Return True if fname_token fuzzy-matches any word in the query."""
            q_tokens = re.split(r'\W+', query_lower)
            for qt in q_tokens:
                if not qt:
                    continue
                # Direct substring containment (Alex ↔ Alexander)
                if fname_token in qt or qt in fname_token:
                    return True
                # Fuzzy similarity for same-length names (MacDonald ↔ McDonald)
                ratio = SequenceMatcher(None, fname_token, qt).ratio()
                if ratio >= 0.82:
                    return True
            return False

        # Load all filenames for this conversation
        file_name_records = list(db["filePages"].find(
            {"conversationId": conversation_id},
            {"fileId": 1, "fileName": 1},
        ))
        query_lower = query.lower()
        for fr in file_name_records:
            fname = fr.get("fileName", "")
            fid = fr.get("fileId")
            if not fname or not fid:
                continue
            # Strip extension and common stopwords for matching
            fname_clean = re.sub(
                r'\.pdf$|\.docx?$|\b(dd|the|and|for|of|in|to|a|an)\b',
                ' ', fname, flags=re.IGNORECASE,
            ).lower()
            # Extract meaningful tokens (3+ chars) — "lau", "kau", "amy" are 3-char
            # surnames/names that were previously excluded by the >= 4 threshold,
            # causing Annie Lau and Carmen Kau files to never trigger the injection.
            tokens = [t for t in re.split(r'\W+', fname_clean) if len(t) >= 3]
            if not tokens:
                continue
            # If 2+ filename tokens fuzzy-match the query, inject the whole file
            hits = sum(1 for t in tokens if _fuzzy_token_match(t, query_lower))
            if hits >= 2:
                hint_sections = list(db["fileSections"].find({
                    "conversationId": conversation_id,
                    "fileId": fid,
                }))
                added_hint = 0
                for sec in hint_sections:
                    key = (sec["fileId"], sec["sectionId"])
                    if key not in seen:
                        seen.add(key)
                        fmeta = file_meta.get(sec["fileId"], {})
                        expanded.append({
                            "sectionId": sec["sectionId"],
                            "sectionTitle": sec.get("sectionTitle", ""),
                            "fileId": sec["fileId"],
                            "fileName": sec.get("fileName", ""),
                            "pageNumber": sec.get("pageNumber", 0),
                            "content": sec.get("content", ""),
                            "clauseType": sec.get("clauseType", "other"),
                            "scheduleContext": sec.get("scheduleContext", ""),
                            "riskLevel": sec.get("riskLevel", 1),
                            "riskNote": sec.get("riskNote", ""),
                            "effectiveDate": fmeta.get("effectiveDate"),
                            "documentType": fmeta.get("documentType", ""),
                            "documentRank": fmeta.get("documentRank", 1),
                            "functionalRole": fmeta.get("functionalRole", "standalone"),
                            "functionalRoleRank": fmeta.get("functionalRoleRank", 1),
                            "score": 0.0,
                            "_filename_hint": True,  # immune to Phase 3 pruning
                        })
                        added_hint += 1
                if added_hint:
                    logger.info(
                        "[SectionRetriever] Phase 2: filename-hint injection '%s' "
                        "added %d section(s) (query token match)",
                        fname, added_hint,
                    )

    # --- Organization entity injection ---
    # When the query mentions an organization name known in entityIndex, inject all
    # sections from every document where that organization appears.
    # Solves "what is Eames Consulting's role?" where the company name exists in
    # document content (and was extracted at parse time) but does NOT appear in any
    # filename, so the filename-hint injector above never fires for it.
    if query:
        from src.services.entity_extractor import get_entity_summary as _get_ent_summary
        ent_summary = _get_ent_summary(conversation_id)
        known_orgs = {
            k.replace("organization:", "").strip()
            for k in ent_summary
            if k.startswith("organization:")
        }
        _query_lower_org = query.lower()
        for org_name in known_orgs:
            if not org_name or len(org_name) < 3:
                continue
            # Match if any meaningful token (3+ chars) of the org name appears in the query
            org_tokens = [t for t in re.split(r'\W+', org_name.lower()) if len(t) >= 3]
            if not org_tokens:
                continue
            if not any(t in _query_lower_org for t in org_tokens):
                continue
            # Find all files that mention this organization via the entity index
            from src.services.entity_extractor import find_files_by_entity as _find_by_ent
            org_file_ids = _find_by_ent(conversation_id, org_name, entity_type="organization")
            for fid in org_file_ids:
                org_sections = list(db["fileSections"].find({
                    "conversationId": conversation_id,
                    "fileId": fid,
                }))
                added_org = 0
                for sec in org_sections:
                    key = (sec["fileId"], sec["sectionId"])
                    if key not in seen:
                        seen.add(key)
                        fmeta = file_meta.get(sec["fileId"], {})
                        expanded.append({
                            "sectionId": sec["sectionId"],
                            "sectionTitle": sec.get("sectionTitle", ""),
                            "fileId": sec["fileId"],
                            "fileName": sec.get("fileName", ""),
                            "pageNumber": sec.get("pageNumber", 0),
                            "content": sec.get("content", ""),
                            "clauseType": sec.get("clauseType", "other"),
                            "scheduleContext": sec.get("scheduleContext", ""),
                            "riskLevel": sec.get("riskLevel", 1),
                            "riskNote": sec.get("riskNote", ""),
                            "effectiveDate": fmeta.get("effectiveDate"),
                            "documentType": fmeta.get("documentType", ""),
                            "documentRank": fmeta.get("documentRank", 1),
                            "functionalRole": fmeta.get("functionalRole", "standalone"),
                            "functionalRoleRank": fmeta.get("functionalRoleRank", 1),
                            "score": 0.0,
                            "_org_entity_injection": True,  # immune to Phase 3 pruning
                        })
                        added_org += 1
                if added_org:
                    logger.info(
                        "[SectionRetriever] Phase 2: org-entity injection '%s' "
                        "added %d section(s) from file %s",
                        org_name, added_org, fid,
                    )

    # --- Physical address injection ---
    # When the query asks about place of service, location, office, or physical address,
    # inject sections containing address-like content from seed files.
    # Solves the failure mode where Section 8 (Services / Place of Service with a specific
    # office address) is a new clause in the 2024 Work Order that Section 3 (Days/Hours)
    # already partially satisfies — so the navigator prunes Section 8 even though the
    # full answer (office address) lives there.
    _LOCATION_QUERY_RE = re.compile(
        r'\b(place\s+of\s+service|locat|address|office|physical|building|floor|premises'
        r'|where|work\s+from|based)\b',
        re.IGNORECASE,
    )
    _ADDRESS_CONTENT_RE = re.compile(
        r'\b(\d+/F|\d+th\s+floor|floor\s+\d+|building|road|street|avenue|drive|lane'
        r'|plaza|tower|centre|center|office\s+at|located\s+at)\b',
        re.IGNORECASE,
    )
    if query and _LOCATION_QUERY_RE.search(query):
        addr_file_ids = list({s["fileId"] for s in seed_sections}) or None
        addr_query: dict = {"conversationId": conversation_id}
        if addr_file_ids:
            addr_query["fileId"] = {"$in": addr_file_ids}
        addr_sections = list(db["fileSections"].find(addr_query, limit=200))
        added_addr = 0
        for sec in addr_sections:
            content = sec.get("content", "")
            if not _ADDRESS_CONTENT_RE.search(content):
                continue
            key = (sec["fileId"], sec["sectionId"])
            if key in seen:
                continue
            seen.add(key)
            fmeta = file_meta.get(sec["fileId"], {})
            expanded.append({
                "sectionId": sec["sectionId"],
                "sectionTitle": sec.get("sectionTitle", ""),
                "fileId": sec["fileId"],
                "fileName": sec.get("fileName", ""),
                "pageNumber": sec.get("pageNumber", 0),
                "content": content,
                "clauseType": sec.get("clauseType", "other"),
                "scheduleContext": sec.get("scheduleContext", ""),
                "riskLevel": sec.get("riskLevel", 1),
                "riskNote": sec.get("riskNote", ""),
                "effectiveDate": fmeta.get("effectiveDate"),
                "documentType": fmeta.get("documentType", ""),
                "documentRank": fmeta.get("documentRank", 1),
                "functionalRole": fmeta.get("functionalRole", "standalone"),
                "functionalRoleRank": fmeta.get("functionalRoleRank", 1),
                "score": 0.0,
                "_address_injection": True,  # immune to Phase 3 pruning
            })
            added_addr += 1
        if added_addr:
            logger.info(
                "[SectionRetriever] Phase 2: address injection added %d section(s) "
                "containing physical location/address content",
                added_addr,
            )

    # --- Novation retained-liability injection ---
    # When the query contains a specific date AND novation/modification documents are
    # present, inject sections from those documents that contain "retained liability",
    # "before the effective date", "acts or omissions", "release and discharge".
    # Solves "who is liable for an act on DATE X?" where the answer depends on which
    # party held obligations at that point — not who holds them today.
    _DATE_IN_QUERY_RE = re.compile(
        r'\b(\d{1,2}[\s/\-\.]\w+[\s/\-\.]\d{4}|\d{4}[\-/]\d{2}[\-/]\d{2}'
        r'|\b(?:january|february|march|april|may|june|july|august|september'
        r'|october|november|december)\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE,
    )
    _RETAINED_LIABILITY_RE = re.compile(
        r'acts?\s+or\s+omissions?|before\s+the\s+effective\s+date|retained?\s+liabil'
        r'|release\s+and\s+discharge|prior\s+to\s+the\s+effective\s+date'
        r'|occurring\s+before|neglect|default.*before',
        re.IGNORECASE,
    )
    if query and _DATE_IN_QUERY_RE.search(query):
        # Scope to novation / modification documents in this conversation
        novation_files = list(db["filePages"].find(
            {
                "conversationId": conversation_id,
                "functionalRole": {"$in": ["modification", "termination"]},
            },
            {"fileId": 1},
        ))
        novation_file_ids = [r["fileId"] for r in novation_files if r.get("fileId")]
        if novation_file_ids:
            novation_sections = list(db["fileSections"].find({
                "conversationId": conversation_id,
                "fileId": {"$in": novation_file_ids},
            }))
            added_nov = 0
            for sec in novation_sections:
                content = sec.get("content", "")
                if not _RETAINED_LIABILITY_RE.search(content):
                    continue
                key = (sec["fileId"], sec["sectionId"])
                if key in seen:
                    continue
                seen.add(key)
                fmeta = file_meta.get(sec["fileId"], {})
                expanded.append({
                    "sectionId": sec["sectionId"],
                    "sectionTitle": sec.get("sectionTitle", ""),
                    "fileId": sec["fileId"],
                    "fileName": sec.get("fileName", ""),
                    "pageNumber": sec.get("pageNumber", 0),
                    "content": content,
                    "clauseType": sec.get("clauseType", "other"),
                    "scheduleContext": sec.get("scheduleContext", ""),
                    "riskLevel": sec.get("riskLevel", 1),
                    "riskNote": sec.get("riskNote", ""),
                    "effectiveDate": fmeta.get("effectiveDate"),
                    "documentType": fmeta.get("documentType", ""),
                    "documentRank": fmeta.get("documentRank", 1),
                    "functionalRole": fmeta.get("functionalRole", "standalone"),
                    "functionalRoleRank": fmeta.get("functionalRoleRank", 1),
                    "score": 0.0,
                    "_retained_liability_injection": True,  # immune to Phase 3 pruning
                })
                added_nov += 1
            if added_nov:
                logger.info(
                    "[SectionRetriever] Phase 2: retained-liability injection added "
                    "%d section(s) from novation/modification documents",
                    added_nov,
                )

    return _phase2_inject_definitions(conversation_id, expanded, query=query)


def _phase2_inject_definitions(
    conversation_id: str, sections: List[dict], query: str = ""
) -> List[dict]:
    """
    Inject defined-term snippets from filePages.definedTerms.

    Trigger 1 — term appears in already-retrieved section content (original).
    Trigger 2 — term appears in the query itself (NEW): handles "what is the
                 Escrow Agent?" where the definitions section may not have been
                 retrieved at all by Phase 1.

    Loads definedTerms maps from filePages for all files in the conversation.
    Injects at most _MAX_DEFINITION_INJECTIONS definitions per query to keep
    context lean.
    """
    import re as _re
    from src.utils.connection_utils import db

    _MAX_DEFINITION_INJECTIONS = 10

    file_ids = list({s["fileId"] for s in sections})

    # Build a combined {term: {definition, fileId, fileName}} map across all files
    combined_terms: dict = {}
    for fp in db["filePages"].find(
        {"fileId": {"$in": file_ids}, "definedTerms": {"$exists": True}},
        {"fileId": 1, "fileName": 1, "definedTerms": 1},
    ):
        fid = fp["fileId"]
        fname = fp.get("fileName", "")
        for term, snippet in fp.get("definedTerms", {}).items():
            if term not in combined_terms:
                combined_terms[term] = {"definition": snippet, "fileId": fid, "fileName": fname}

    if not combined_terms:
        return sections

    # Sort longest terms first so "Subscription Period" matches before "Period"
    sorted_terms = sorted(combined_terms.keys(), key=len, reverse=True)
    term_pattern = _re.compile(
        r'\b(' + '|'.join(_re.escape(t) for t in sorted_terms) + r')\b'
    )

    injected_terms: set = set()
    injections: list = []
    already_injected = {s.get("sectionId", "") for s in sections}

    # --- Trigger 2 FIRST: inject definitions for terms in the query ---
    # Run before Trigger 1 so that query-specific terms are guaranteed a slot
    # before Trigger 1 (section-content scan) can exhaust the cap with unrelated terms.
    # E.g. "who is the Escrow Agent?" must inject "Escrow Agent" → "Wilmington Trust"
    # even if Trigger 1 already consumed 10 slots with terms from retrieved sections.
    _QUERY_TERM_CAP = 5   # dedicated slots for query-matched terms
    query_injected: set = set()
    if query:
        for term in sorted_terms:
            if len(query_injected) >= _QUERY_TERM_CAP:
                break
            if term in injected_terms:
                continue
            if term.lower() in query.lower():
                synthetic_id = f"__def_{term.lower().replace(' ', '_')}"
                if synthetic_id in already_injected:
                    injected_terms.add(term)  # count as injected so Trigger 1 skips it
                    query_injected.add(term)
                    continue
                info = combined_terms[term]
                injected_terms.add(term)
                query_injected.add(term)
                already_injected.add(synthetic_id)
                injections.append({
                    "sectionId": synthetic_id,
                    "sectionTitle": f'Definition: "{term}"',
                    "fileId": info["fileId"],
                    "fileName": info["fileName"],
                    "pageNumber": 0,
                    "content": f'"{term}" means {info["definition"]}',
                    "score": 0.0,
                    "is_definition_injection": True,
                    "_definition_lookup": True,  # immune to Phase 3 pruning
                })

    # --- Trigger 1: inject definitions for terms in retrieved section content ---
    for sec in sections:
        if len(injected_terms) >= _MAX_DEFINITION_INJECTIONS:
            break
        for match in term_pattern.finditer(sec.get("content", "")):
            term = match.group(1)
            if term in injected_terms:
                continue
            synthetic_id = f"__def_{term.lower().replace(' ', '_')}"
            if synthetic_id in already_injected:
                continue
            info = combined_terms[term]
            injected_terms.add(term)
            already_injected.add(synthetic_id)
            injections.append({
                "sectionId": synthetic_id,
                "sectionTitle": f'Definition: "{term}"',
                "fileId": info["fileId"],
                "fileName": info["fileName"],
                "pageNumber": 0,
                "content": f'"{term}" means {info["definition"]}',
                "score": 0.0,
                "is_definition_injection": True,
            })
            if len(injected_terms) >= _MAX_DEFINITION_INJECTIONS:
                break

    if injections:
        logger.info(
            "[SectionRetriever] Phase 2: injected %d definition(s): %s",
            len(injections),
            ", ".join(f'"{t}"' for t in injected_terms),
        )

    # Targeted definition scan: when query looks like "what is X" / "who is X" and X
    # was NOT already resolved via definedTerms (e.g. quote-character mismatch during
    # extraction), scan the raw definitions section content for lines containing the
    # subject term and inject only those matching lines.
    #
    # This avoids flooding context with the entire definitions section (which can be
    # 8+ pages) while still surfacing "Escrow Agent means Wilmington Trust" even when
    # that entry was missed by _extract_defined_terms at parse time.
    if query:
        _DEF_QUERY_RE = _re.compile(
            r'\b(what\s+is|what\s+are|who\s+is|define|definition\s+of|meaning\s+of)\b',
            _re.IGNORECASE,
        )
        if _DEF_QUERY_RE.search(query):
            # Strip question-word stopwords to get the subject term
            subject = _re.sub(
                r'\b(what\s+is|what\s+are|who\s+is|define|definition\s+of|meaning\s+of'
                r'|the|a|an|in|of|this|that|agreement|contract|please|tell|me)\b',
                ' ',
                query,
                flags=_re.IGNORECASE,
            ).strip(' \t\n?.,')
            subject_lower = subject.lower().strip()

            if len(subject_lower) >= 3:
                def_sections = list(db["fileSections"].find({
                    "conversationId": conversation_id,
                    "clauseType": "definitions",
                }, limit=5))

                for sec in def_sections:
                    content = sec.get("content", "")
                    if not content or subject_lower not in content.lower():
                        continue
                    # Collect lines that mention the subject term
                    matching_lines = [
                        ln for ln in content.split('\n')
                        if subject_lower in ln.lower()
                    ]
                    if not matching_lines:
                        continue
                    snippet = '\n'.join(matching_lines[:15])  # cap at 15 lines
                    safe_key = subject_lower[:40].replace(' ', '_')
                    synthetic_id = f"__def_scan_{safe_key}"
                    if synthetic_id in already_injected:
                        continue
                    already_injected.add(synthetic_id)
                    injections.append({
                        "sectionId": synthetic_id,
                        "sectionTitle": f'Definition (scan): "{subject[:60]}"',
                        "fileId": sec["fileId"],
                        "fileName": sec.get("fileName", ""),
                        "pageNumber": sec.get("pageNumber", 0),
                        "content": snippet,
                        "clauseType": "definitions",
                        "score": 0.0,
                        "is_definition_injection": True,  # renders as [Background definition]
                        "_definition_lookup": True,       # immune to Phase 3 pruning
                    })
                    logger.info(
                        "[SectionRetriever] Phase 2: definition scan injected %d line(s) "
                        "for subject '%s'",
                        len(matching_lines), subject[:60],
                    )
                    break  # one definitions section is enough per conversation

    return sections + injections


# ---------------------------------------------------------------------------
# Phase 3: Cross-encoder re-ranking (replaces LLM navigation)
# ---------------------------------------------------------------------------

# Shared regex for comparison/progression queries — used in both Phase 3 and
# any future phase that needs to detect multi-document comparison intent.
_COMPARISON_QUERY_RE = re.compile(
    r'\b(compare|comparison|versus|vs\.?|track|progression|history|increase|change'
    r'|evolv|earliest|most\s+recent|over\s+time|between.*and|all.*work\s+order'
    r'|each.*work\s+order|across.*document|version\s+histor)\b',
    re.IGNORECASE,
)


def _phase3_llm_navigate(query: str, candidates: List[dict]) -> List[dict]:
    """
    Phase 3: cross-encoder re-ranking.

    Scores every (query, section) pair with a local cross-encoder model
    (cross-encoder/ms-marco-MiniLM-L-6-v2) and keeps the top-ranked sections.
    Replaces the previous Gemini LLM navigation call — same logic, zero API cost,
    ~20-50 ms on CPU, no rate limits, no JSON parsing.

    Immune sections (financial injections, structural markers, governing law, etc.)
    always pass through regardless of score, identical to the previous behaviour.

    Falls back to all candidates if the model is not loaded or errors.
    """
    if len(candidates) <= 5:
        return candidates

    # Sections that were already precision-filtered upstream pass through unconditionally.
    def _is_immune(s: dict) -> bool:
        return bool(
            s.get("_financial_injection")
            or s.get("_filename_hint")
            or s.get("_retained_liability_injection")
            or s.get("_address_injection")
            or s.get("_wo_structural")
            or s.get("_definition_lookup")
            or s.get("_preamble_injection")
            or s.get("_governing_law_injection")
            or s.get("_dispute_resolution_injection")
            or s.get("_sla_injection")
            or s.get("_org_entity_injection")
            or s.get("_graph_injection")   # graph-traversal injections always pass
        )

    always_pass = [s for s in candidates if _is_immune(s)]
    nav_candidates = [s for s in candidates if not _is_immune(s)]

    if not nav_candidates:
        return candidates

    # Increase limit for comparison/progression queries so sections from every
    # document version survive (e.g. fee tracking across 3 Work Orders).
    nav_limit = 25 if _COMPARISON_QUERY_RE.search(query) else 20

    def _section_text(s: dict) -> str:
        """Build the passage text the cross-encoder scores against the query."""
        title = s.get("sectionTitle", "").strip()
        ct = s.get("clauseType", "")
        content = s.get("content", "").strip().replace("\n", " ")[:400]
        parts = []
        if title:
            parts.append(title)
        if ct and ct not in ("other", "general", ""):
            parts.append(f"[{ct}]")
        if content:
            parts.append(content)
        return " | ".join(parts) if parts else content

    try:
        from src.utils.cross_encoder_instance import get_cross_encoder
        ce = get_cross_encoder()

        pairs = [(query, _section_text(s)) for s in nav_candidates]
        scores = ce.predict(pairs)  # numpy array, higher = more relevant

        scored = sorted(zip(scores, nav_candidates), key=lambda x: x[0], reverse=True)

        seen_keys: set = {(s["fileId"], s["sectionId"]) for s in always_pass}
        selected = []
        for _score, sec in scored[:nav_limit]:
            key = (sec["fileId"], sec["sectionId"])
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append(sec)

        confirmed = always_pass + selected
        logger.info(
            f"[SectionRetriever] Phase 3 (cross-encoder): {len(always_pass)} immune + "
            f"{len(selected)}/{len(nav_candidates)} re-ranked → {len(confirmed)} total"
        )
        return confirmed

    except Exception as e:
        logger.warning(f"[SectionRetriever] Phase 3 cross-encoder failed, using all candidates: {e}")

    return candidates


# ---------------------------------------------------------------------------
# Phase 3.5: Gap analysis — self-correcting section fetch
# ---------------------------------------------------------------------------

def _phase35_gap_analysis(
    query: str,
    confirmed: List[dict],
    conversation_id: str,
) -> List[dict]:
    """
    Show the LLM the evidence it already has (headings + content snippets) and
    ask it to name up to 5 specific additional sections needed to fully answer
    the query.  The system then fetches those sections from MongoDB.

    Sections are identified by clause reference (e.g. "§4.3", "Schedule 7 §2")
    or title fragment.  Returns an empty list if the LLM is satisfied or if the
    call fails.
    """
    if not confirmed:
        return []

    from src.utils.llm_utils import invoke_with_costing_evalution
    from src.utils.connection_utils import db

    # Build a compact evidence summary for the LLM
    evidence_lines = []
    for s in confirmed:
        schedule = s.get("scheduleContext", "")
        sid = s["sectionId"]
        ref = f"{schedule.title()} §{sid}" if schedule and sid != schedule else f"§{sid}"
        title = s.get("sectionTitle", "")
        content = s.get("content", "").strip().replace("\n", " ")[:100]
        evidence_lines.append(f"  {ref} — {title} | {content}…")

    evidence_text = "\n".join(evidence_lines)

    prompt = (
        "You are a legal analyst reviewing contract evidence.\n\n"
        f"Question: {query}\n\n"
        "Evidence already retrieved:\n"
        f"{evidence_text}\n\n"
        "Identify up to 5 specific additional sections you need to fully answer the "
        "question. Name them by clause number (e.g. '§4.3', 'Schedule 7 §2.1') or "
        "section title fragment. If you have sufficient evidence, return [].\n\n"
        "Output ONLY a JSON array of strings.\n"
        'Examples: ["§4.3", "Schedule 7 §2", "Liability Cap"]  or  []'
    )

    try:
        response = invoke_with_costing_evalution(prompt=prompt)
        raw = response.content.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end <= start:
            return []

        gap_refs = json.loads(raw[start:end])
        if not isinstance(gap_refs, list) or not gap_refs:
            return []

        logger.info(
            "[SectionRetriever] Phase 3.5: LLM requested %d additional section(s): %s",
            len(gap_refs), gap_refs,
        )

        seen_keys = {(s["fileId"], s["sectionId"]) for s in confirmed}
        additions = []

        for ref in gap_refs[:5]:
            ref_str = str(ref).strip()
            if not ref_str:
                continue

            # Match by sectionId — extract digit-and-dot pattern (e.g. "4.3" from "§4.3")
            sid_match = re.search(r'(\d[\d\.]*\d|\d)', ref_str)
            if sid_match:
                sid_pattern = sid_match.group(1).replace(".", r"\.")
                candidates = list(db["fileSections"].find({
                    "conversationId": conversation_id,
                    "sectionId": re.compile(rf'^{sid_pattern}$'),
                }, limit=5))
                for sec in candidates:
                    key = (sec["fileId"], sec["sectionId"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        additions.append({
                            "sectionId": sec["sectionId"],
                            "sectionTitle": sec.get("sectionTitle", ""),
                            "fileId": sec["fileId"],
                            "fileName": sec.get("fileName", ""),
                            "pageNumber": sec.get("pageNumber", 0),
                            "content": sec.get("content", ""),
                            "clauseType": sec.get("clauseType", "other"),
                            "scheduleContext": sec.get("scheduleContext", ""),
                            "riskLevel": sec.get("riskLevel", 1),
                            "riskNote": sec.get("riskNote", ""),
                            "score": 0.0,
                            "_gap_injection": True,
                        })

            # Also match by title fragment (strip §/digits to get the words)
            title_words = re.sub(r'[§\d\.\s]+', ' ', ref_str).strip()
            if len(title_words) > 3:
                title_candidates = list(db["fileSections"].find({
                    "conversationId": conversation_id,
                    "sectionTitle": re.compile(re.escape(title_words), re.IGNORECASE),
                }, limit=3))
                for sec in title_candidates:
                    key = (sec["fileId"], sec["sectionId"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        additions.append({
                            "sectionId": sec["sectionId"],
                            "sectionTitle": sec.get("sectionTitle", ""),
                            "fileId": sec["fileId"],
                            "fileName": sec.get("fileName", ""),
                            "pageNumber": sec.get("pageNumber", 0),
                            "content": sec.get("content", ""),
                            "clauseType": sec.get("clauseType", "other"),
                            "scheduleContext": sec.get("scheduleContext", ""),
                            "riskLevel": sec.get("riskLevel", 1),
                            "riskNote": sec.get("riskNote", ""),
                            "score": 0.0,
                            "_gap_injection": True,
                        })

        if additions:
            logger.info(
                "[SectionRetriever] Phase 3.5: fetched %d gap section(s)", len(additions)
            )
        return additions

    except Exception as e:
        logger.warning("[SectionRetriever] Phase 3.5 gap analysis failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Phase 4: Assemble context text + citation metadata
# ---------------------------------------------------------------------------

def _phase4_assemble(sections: List[dict], file_meta: Dict) -> Tuple[str, Dict]:
    """
    Format section content as [Source N: filename, Page X] blocks and build
    the citation_metadata dict — same structure as load_full_documents_with_citations.

    Ordering (specific-over-general, most-recent-first within same tier):
      1. Schedule/Appendix provisions (authoritative specific overrides)
      2. Most recent document first by effectiveDate (amendments before originals)
      3. Document rank — higher rank = more engagement-specific (work_order=2 beats framework=0)
      4. Semantic score as final tie-breaker

    Temporal annotation: when the same clauseType appears in multiple documents,
    sections from the older document are prefixed [HISTORICAL VERSION] so the LLM
    knows the newer provision governs.
    """
    from collections import defaultdict
    context_parts = []
    citation_metadata: Dict = {}
    source_index = 1

    # Identify which file holds the CURRENT (most recent) version of each clauseType
    # that appears in more than one file.  Only meaningful clauseTypes considered.
    _CONFLICT_TYPES = {
        "payment_terms", "liability", "limitation_of_liability", "sla_uptime",
        "termination", "term_and_termination", "indemnification", "data_protection",
        "confidentiality", "warranties", "intellectual_property",
    }
    clause_type_file_dates: defaultdict = defaultdict(list)
    for s in sections:
        ct = s.get("clauseType", "other")
        if ct in _CONFLICT_TYPES and not s.get("is_definition_injection"):
            clause_type_file_dates[ct].append(
                (s.get("effectiveDate") or "0000-00-00", s["fileId"])
            )

    latest_file_for_type: Dict[str, str] = {}
    for ct, entries in clause_type_file_dates.items():
        file_ids = {e[1] for e in entries}
        if len(file_ids) > 1:
            # Primary: most recent effectiveDate wins.
            # Secondary (Fix 2): when dates are equal or all None, highest
            # documentRank wins (transaction=2 beats framework=0).
            has_real_dates = any(e[0] != "0000-00-00" for e in entries)
            if has_real_dates:
                latest_file_for_type[ct] = max(entries, key=lambda e: e[0])[1]
            else:
                # All dates are absent — use documentRank as tiebreaker
                rank_entries = [
                    (s.get("documentRank", 1), s["fileId"])
                    for s in sections
                    if s.get("clauseType") == ct and not s.get("is_definition_injection")
                ]
                if rank_entries:
                    latest_file_for_type[ct] = max(rank_entries, key=lambda e: e[0])[1]

    # Sort: schedules first → most recent date → highest functional role rank
    # (transaction > modification > master_agreement) → semantic score
    # functionalRoleRank: 0=master, 1=modification, 2=transaction, 3=termination
    # Higher rank = more specific = should appear first so LLM sees specific override before general rule
    def _section_sort_key(s: dict):
        is_schedule = 1 if s.get("scheduleContext") else 0
        date_str = s.get("effectiveDate") or "0000-00-00"
        role_rank = s.get("functionalRoleRank", 1)
        score = s.get("score", 0.0)
        # Negate date_str for descending; higher role_rank first; higher score first
        return (-is_schedule, tuple(~ord(c) for c in date_str), -role_rank, -score)

    ordered = sorted(sections, key=_section_sort_key)

    for sec in ordered:
        content = sec.get("content", "").strip()
        if not content:
            continue
        file_name = sec.get("fileName", "Unknown")
        page_num = sec.get("pageNumber", "?")
        schedule = sec.get("scheduleContext", "")
        sid = sec.get("sectionId", "")

        if sec.get("is_definition_injection"):
            context_parts.append(f"[Background definition]\n{content}\n")
            continue

        # Build clause reference — suppress synthetic internal IDs (start with __)
        if schedule and sid != schedule:
            clause_ref = f", {schedule.title()} §{sid}"
        elif sid and not sid.startswith("__"):
            clause_ref = f", §{sid}"
        else:
            clause_ref = ""

        # Temporal annotation
        effective = sec.get("effectiveDate", "")
        doc_type = sec.get("documentType", "")
        temporal_tag = ""
        if effective:
            temporal_tag = f" [effective {effective}]"

        ct = sec.get("clauseType", "other")
        historical_tag = ""
        if ct in latest_file_for_type and latest_file_for_type[ct] != sec["fileId"]:
            historical_tag = " [HISTORICAL VERSION — superseded by more recent document]"

        # Risk annotation for high-risk sections
        risk_tag = ""
        risk_level = sec.get("riskLevel", 1)
        risk_note = sec.get("riskNote", "")
        if risk_level >= 4:
            risk_label = "HIGH RISK" if risk_level == 5 else "ABOVE MARKET"
            risk_tag = f" [⚠ {risk_label}: {risk_note}]" if risk_note else f" [⚠ {risk_label}]"

        header = (
            f"[Source {source_index}: {file_name}{clause_ref}, "
            f"Page {page_num}{temporal_tag}{historical_tag}{risk_tag}]"
        )
        context_parts.append(f"{header}\n{content}\n")
        citation_metadata[source_index] = {
            "fileId": sec["fileId"],
            "fileName": file_name,
            "pageNumber": str(page_num),
            "effectiveDate": effective,
            "documentType": doc_type,
            "riskLevel": risk_level,
        }
        source_index += 1

    logger.info(
        f"[SectionRetriever] Phase 4: assembled {source_index - 1} source(s); "
        f"{len(latest_file_for_type)} clauseType conflict(s) annotated"
    )
    return "\n".join(context_parts), citation_metadata


# ---------------------------------------------------------------------------
# Phase 4b: Table normalization
# ---------------------------------------------------------------------------

def _phase4b_normalize_tables(context_text: str) -> str:
    """
    Detect pipe-delimited or tab-aligned tables in the assembled context and
    use the LLM to reformat each table as clear 'Key: Value' lines.

    Only fires when the context contains both table patterns AND financial/SLA
    values (currency amounts or percentages), targeting the specific failure
    mode where £90,000 or 99.9% sit in a table column that the answer LLM
    skips because the alignment is ambiguous.

    Non-table text is passed through unchanged.  Skipped entirely if the
    context has no tables, keeping latency near-zero for most queries.
    """
    from src.utils.llm_utils import invoke_with_costing_evalution

    _TABLE_RE = re.compile(r'\|.*\||\t[^\t]+\t')
    _VALUE_RE = re.compile(r'(?:£|€|\$)[\d,]+|\d+(?:\.\d+)?\s*%')

    if not _TABLE_RE.search(context_text):
        return context_text
    if not _VALUE_RE.search(context_text):
        return context_text

    # Cap input to avoid huge prompts; preserve the tail verbatim
    cap = 3000
    head = context_text[:cap]
    tail = context_text[cap:] if len(context_text) > cap else ""

    prompt = (
        "The following is extracted contract text that contains one or more tables. "
        "Reformat ONLY the table rows as clear 'Key: Value' lines (one per row). "
        "Keep all non-table text exactly as-is. Do not summarize, add commentary, "
        "or omit any values — every number, amount, and percentage must be preserved.\n\n"
        f"{head}"
    )

    try:
        response = invoke_with_costing_evalution(prompt=prompt)
        normalized = response.content.strip()
        if normalized:
            logger.info("[SectionRetriever] Phase 4b: table normalization applied")
            return normalized + ("\n" + tail if tail else "")
    except Exception as e:
        logger.warning("[SectionRetriever] Phase 4b table normalization failed: %s", e)

    return context_text
