"""
Cross-reference extractor and resolver for legal contracts.

extract_cross_references(file_id, conversation_id)
    Scans fileSections for a given file, finds all outgoing clause/section/
    schedule references in each section's content, and writes them to the
    fileCrossRefs collection.

resolve_cross_references(conversation_id)
    For every unresolved fileCrossRef in the conversation, looks for a
    matching fileSections document and records resolvedFileId +
    resolvedSectionId.  Safe to call multiple times — only touches records
    where resolvedFileId is still None.

fetch_resolved_sections(conversation_id, section_ids, file_id=None)
    Convenience query used by full_document_loader at query time.

get_cross_refs_for_sections(conversation_id, section_ids, source_file_id=None)
    Returns all resolved cross-refs whose sourceSectionId is in the list.
    Used for depth-2 expansion in full_document_loader.
"""
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared regex
# ---------------------------------------------------------------------------

# Matches "clause 12.1", "Section 4.2(b)", "Schedule 4", "paragraph 3.1"
_CLAUSE_REF_RE = re.compile(
    r'\b((?:clause|section|paragraph)\s+\d+(?:\.\d+)*(?:\([a-z]{1,2}\))*'
    r'|(?:schedule|appendix|exhibit)\s+\d+)',
    re.IGNORECASE,
)

# Matches "of the Framework Agreement", "of the SaaS Agreement",
# "of the Initial Addendum", "of the SOW"
_DOC_HINT_RE = re.compile(
    r'of\s+(?:the\s+)?([\w\s]+?(?:Agreement|Addendum|Schedule|SOW))\b',
    re.IGNORECASE,
)

_STOPWORDS = {
    'agreement', 'the', 'of', 'and', 'for', 'an', 'a', 'to', 'initial', 'this',
}

# Bare clause numbers above this threshold are almost certainly page/line
# numbers picked up from TOC text (e.g. "clause 26.....14"), not real
# clause cross-references.  Legal contracts rarely exceed 50 top-level clauses.
_MAX_BARE_CLAUSE_NUMBER = 50


def _derive_parent_ref(ref_clause: str) -> Optional[str]:
    """
    Strip the last segment from a dotted clause reference to derive the parent.

    "12.1"   → "12"
    "12.1.2" → "12.1"
    "4.2(b)" → "4.2"
    "12"     → None  (already top-level, no parent to fall back to)
    "schedule 4" → None  (non-numeric, no parent)
    """
    # Non-numeric identifiers (schedule, appendix, etc.) have no numeric parent
    if re.match(r'^(?:schedule|appendix|exhibit|article)', ref_clause, re.IGNORECASE):
        return None
    # Strip trailing sub-item like "(b)" → base without sub-item
    m = re.search(r'\([a-z]{1,2}\)$', ref_clause)
    if m:
        parent = ref_clause[:m.start()]
        return parent if parent else None
    parts = ref_clause.split('.')
    if len(parts) > 1:
        return '.'.join(parts[:-1])
    return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_ref(raw: str) -> str:
    """
    Strip the leading keyword from a cross-reference string to produce the
    canonical sectionId used in fileSections lookups.

    "clause 12.1"    → "12.1"
    "Section 4.2(b)" → "4.2(b)"
    "Schedule 4"     → "schedule 4"
    "paragraph 3.1"  → "3.1"
    """
    m = re.match(
        r'^(?:clause|section|paragraph)\s+(.+)$',
        raw.strip(),
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip('.')

    m = re.match(
        r'^((?:schedule|appendix|exhibit)\s+\d+)',
        raw.strip(),
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().lower()

    return raw.strip().rstrip('.')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_refs_from_text(text: str) -> List[tuple]:
    """
    Return a deduplicated list of (normalised_ref, doc_hint_or_None) tuples
    found in *text*.
    """
    results = []
    seen = set()

    for m in _CLAUSE_REF_RE.finditer(text):
        raw = m.group(0)
        norm = normalize_ref(raw)
        if norm in seen:
            continue

        # Filter out bare integers that are likely page/line numbers from the
        # TOC (e.g. "clause 205" or "clause 26" extracted from
        # "3 AGREEMENT STRUCTURE..........26").
        # Also skip "0" which is never a valid clause reference.
        if norm.isdigit() and (int(norm) == 0 or int(norm) > _MAX_BARE_CLAUSE_NUMBER):
            continue

        seen.add(norm)

        # Look at the 100 chars right after the match for a document name hint
        after = text[m.end():m.end() + 100]
        hint_m = _DOC_HINT_RE.search(after)
        doc_hint = hint_m.group(1).strip() if hint_m else None
        results.append((norm, doc_hint))

    return results


def _files_matching_hint(hint: str, files: list) -> list:
    """
    Return file dicts ordered by keyword overlap with the document name hint.
    Falls back to all files when no useful overlap is found.
    """
    hint_words = set(re.findall(r'\w+', hint.lower())) - _STOPWORDS
    if not hint_words:
        return files

    scored = []
    for f in files:
        fname_words = (
            set(re.findall(r'\w+', f.get('fileName', '').lower())) - _STOPWORDS
        )
        overlap = hint_words & fname_words
        if overlap:
            scored.append((len(overlap), f))

    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored] if scored else files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_cross_references(file_id: str, conversation_id: str):
    """
    Scan all fileSections for *file_id* and write outgoing cross-references
    to fileCrossRefs.  Replaces any existing entries for this file.
    """
    from src.utils.connection_utils import db
    from src.utils.log_utils import insert_logs

    try:
        sections = list(db['fileSections'].find({'fileId': file_id}))
        if not sections:
            logger.warning(
                f"[CrossRefExtractor] No sections for fileId={file_id}; skipping"
            )
            return

        # Drop old cross-ref entries for this file before re-inserting
        db['fileCrossRefs'].delete_many({'sourceFileId': file_id})

        records = []
        for sec in sections:
            section_id = sec.get('sectionId', '')
            content = sec.get('content', '')
            refs = _extract_refs_from_text(content)

            for norm_ref, doc_hint in refs:
                if norm_ref == section_id:
                    continue  # skip self-reference
                records.append({
                    'conversationId': conversation_id,
                    'sourceFileId': file_id,
                    'sourceSectionId': section_id,
                    'refClause': norm_ref,
                    'refDocHint': doc_hint,
                    'resolvedFileId': None,
                    'resolvedSectionId': None,
                })

        if records:
            db['fileCrossRefs'].insert_many(records)

        insert_logs(
            message=(
                f"[CrossRefExtractor] Extracted {len(records)} cross-ref(s) "
                f"from fileId={file_id}"
            ),
            logType='information',
            bulkId=file_id,
        )

    except Exception as e:
        logger.error(
            f"[CrossRefExtractor] extract_cross_references failed "
            f"for fileId={file_id}: {e}"
        )


def _resolve_ref_in_files(
    db,
    ref_clause: str,
    candidate_file_ids: list,
) -> Optional[tuple]:
    """
    Try to resolve *ref_clause* against fileSections in the given file order.

    Resolution strategy (in order):
      1. Exact sectionId match  — "8.2"  → sectionId "8.2"
      2. Child fallback          — "8.2"  → first stored child whose parentSectionId
                                             is "8.2" (e.g. "8.2.1").  Handles
                                             documents that store only sub-clauses
                                             (the parent header line is too short to
                                             survive the _MIN_CONTENT_CHARS filter).
                                             Also covers bare integers: "8" → "8.1".

    Returns (resolvedFileId, resolvedSectionId) or None.
    """
    for fid in candidate_file_ids:
        # 1. Exact match
        hit = db['fileSections'].find_one({'fileId': fid, 'sectionId': ref_clause})
        if hit:
            return fid, ref_clause

        # 2. Child fallback — works for both "8" → "8.1" and "8.2" → "8.2.1"
        child = db['fileSections'].find_one(
            {'fileId': fid, 'parentSectionId': ref_clause},
            sort=[('sectionId', 1)],
        )
        if child:
            return fid, child['sectionId']

    # 3. Parent fallback — "12.1" not found anywhere → try parent "12".
    #    Strips the last dotted segment (e.g. "12.1" → "12", "4.2(b)" → "4.2").
    #    Ensures the caller still gets a meaningful section rather than nothing.
    parent = _derive_parent_ref(ref_clause)
    if parent:
        for fid in candidate_file_ids:
            hit = db['fileSections'].find_one({'fileId': fid, 'sectionId': parent})
            if hit:
                return fid, parent

    return None


def resolve_cross_references(conversation_id: str):
    """
    Attempt to resolve all unresolved fileCrossRefs in the conversation by
    matching each refClause to a fileSections document.

    Safe to call multiple times — only processes records where
    resolvedFileId is None.

    Failure reasons logged per-ref:
      no_dot      — bare integer like "8"; bare-number fallback also tried
      not_found   — dotted ID not present in any file's sections
    """
    from src.utils.connection_utils import db
    from src.utils.db_utils import get_fileInfo_from_conversation
    from src.utils.log_utils import insert_logs

    try:
        unresolved = list(db['fileCrossRefs'].find({
            'conversationId': conversation_id,
            'resolvedFileId': None,
        }))

        if not unresolved:
            return

        files = get_fileInfo_from_conversation(conversation_id) or []
        resolved_count = 0
        failures: dict = {}  # reason → list of refClause strings

        for ref in unresolved:
            ref_clause = ref.get('refClause', '')
            doc_hint = ref.get('refDocHint')
            source_file_id = ref.get('sourceFileId')

            # Determine candidate file order
            if doc_hint:
                candidate_file_ids = [
                    f['fileId']
                    for f in _files_matching_hint(doc_hint, files)[:3]
                ]
            else:
                # Within-document first, then other files
                candidate_file_ids = [source_file_id] + [
                    f['fileId']
                    for f in files
                    if f['fileId'] != source_file_id
                ]

            result = _resolve_ref_in_files(db, ref_clause, candidate_file_ids)

            if result:
                resolved_fid, resolved_sid = result
                db['fileCrossRefs'].update_one(
                    {'_id': ref['_id']},
                    {'$set': {
                        'resolvedFileId': resolved_fid,
                        'resolvedSectionId': resolved_sid,
                    }},
                )
                resolved_count += 1
            else:
                failures.setdefault('not_found', []).append(ref_clause)

        # Summary log
        insert_logs(
            message=(
                f"[CrossRefResolver] Resolved {resolved_count}/{len(unresolved)} "
                f"cross-ref(s) for conversation {conversation_id}"
            ),
            logType='information',
            bulkId=conversation_id,
        )

        # Diagnostic breakdown — only logged when failures exist
        if failures:
            for reason, clauses in failures.items():
                unique = sorted(set(clauses))
                logger.warning(
                    f"[CrossRefResolver] Unresolved ({reason}) x{len(clauses)}: "
                    f"{', '.join(unique[:20])}"
                    + (' …' if len(unique) > 20 else '')
                )

    except Exception as e:
        logger.error(
            f"[CrossRefExtractor] resolve_cross_references failed "
            f"for conversation={conversation_id}: {e}"
        )


def fetch_resolved_sections(
    conversation_id: str,
    section_ids: List[str],
    file_id: Optional[str] = None,
) -> List[dict]:
    """
    Fetch fileSections documents matching the given section IDs.

    Used by full_document_loader at query time to pull in cross-referenced
    sections with real page numbers.
    """
    from src.utils.connection_utils import db

    if not section_ids:
        return []

    query: dict = {
        'conversationId': conversation_id,
        'sectionId': {'$in': section_ids},
    }
    if file_id:
        query['fileId'] = file_id

    return list(db['fileSections'].find(query))


def get_cross_refs_for_sections(
    conversation_id: str,
    section_ids: List[str],
    source_file_id: Optional[str] = None,
) -> List[dict]:
    """
    Return all *resolved* fileCrossRefs whose sourceSectionId is in
    section_ids.  Used for depth-2 chain expansion in full_document_loader.
    """
    from src.utils.connection_utils import db

    if not section_ids:
        return []

    query: dict = {
        'conversationId': conversation_id,
        'sourceSectionId': {'$in': section_ids},
        'resolvedFileId': {'$ne': None},
    }
    if source_file_id:
        query['sourceFileId'] = source_file_id

    return list(db['fileCrossRefs'].find(query))
