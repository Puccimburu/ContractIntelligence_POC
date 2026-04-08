"""
Schema-Agnostic Entity Extractor

Reads the first ~800 words of any legal document and uses an LLM to extract
a structured "bag of entities" — regardless of document format, company style,
or field naming convention.

Output stored in two MongoDB collections:

documentEntities (one record per file):
    {
        "fileId": str,
        "conversationId": str,
        "functionalRole": str,       # master_agreement | transaction | modification | termination | standalone
        "persons": [{"name": str, "role": str}],
        "organizations": [{"name": str, "role": str}],
        "financial_terms": [{"key": str, "value": str, "currency": str}],
        "dates": [{"key": str, "value": str}],
        "references": [{"key": str, "value": str}],
        "identifiers": [{"key": str, "value": str}],
    }

entityIndex (one record per entity occurrence, for fast lookup):
    {
        "conversationId": str,
        "fileId": str,
        "entityType": str,    # person | organization | project_code | agreement_ref
        "entityValue": str,   # raw value as extracted
        "entityKey": str,     # normalized lower-case key for matching
    }

Public API:

    extract_entities(file_id, conversation_id) -> dict
        Run extraction for one file. Idempotent — overwrites previous result.

    find_files_by_entity(conversation_id, entity_value) -> list[str]
        Return all fileIds that mention the entity (fuzzy normalized match).

    get_entity_summary(conversation_id) -> dict
        Return {entity_key: [fileId, ...]} for all entities in the conversation.
"""

import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for fuzzy matching."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_ENTITY_EXTRACTION_PROMPT = """You are a legal document analyst. Read the document excerpt below and extract structured entities.

Return ONLY a valid JSON object with this exact schema:

{{
  "functionalRole": "<one of: master_agreement | transaction | modification | termination | standalone>",
  "persons": [{{"name": "<full name>", "role": "<Consultant|Client Representative|Signatory|etc>"}}],
  "organizations": [{{"name": "<full legal entity name>", "role": "<Client|Service Provider|Outgoing Party|Incoming Party|etc>"}}],
  "financial_terms": [{{"key": "<label used in document>", "value": "<amount or formula>", "currency": "<HKD|SGD|GBP|USD|etc or null>"}}],
  "dates": [{{"key": "<label e.g. Commencement Date>", "value": "<date string as written>"}}],
  "references": [{{"key": "<label e.g. Master Agreement>", "value": "<referenced document name and date>"}}],
  "identifiers": [{{"key": "<label e.g. Project Code>", "value": "<value>"}}]
}}

Functional Role definitions:
- master_agreement: Root/hub contract (MSA, Framework Agreement, Master Consultancy Agreement, MCA)
- transaction: Individual engagement under a master (Work Order, SOW, Statement of Work, Purchase Order, Task Order)
- modification: Changes or transfers an existing agreement (Amendment, Novation, Side Letter, Addendum, Variation)
- termination: Ends an agreement (Termination Letter, Notice of Termination, Expiry Notice)
- standalone: Not linked to a hub (NDA, Policy, Standalone License)

Rules:
- Extract ALL persons mentioned (consultants, signatories, representatives)
- Extract ALL organizations (use full legal names where visible)
- financial_terms: include day rates, monthly fees, annual fees, formulas, caps, penalties
- dates: include commencement, expiry, effective, signature dates
- references: name every other agreement explicitly cited
- identifiers: project codes, file numbers, reference codes, batch numbers
- If a field has no entries, return an empty array []
- Do NOT add commentary outside the JSON

Document excerpt:
{document_excerpt}
"""


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _get_excerpt(page_wise_text: Dict[str, str], max_chars: int = 2000) -> str:
    """Return the first max_chars of the document across the first few pages."""
    def _page_int(key: str) -> int:
        return int(re.sub(r'[^0-9]', '', key) or '0')

    sorted_pages = sorted(page_wise_text.items(), key=lambda x: _page_int(x[0]))
    combined = '\n'.join(text for _, text in sorted_pages[:3])
    return combined[:max_chars]


def extract_entities(file_id: str, conversation_id: str) -> dict:
    """
    Extract entities from the document associated with file_id.
    Reads pageWiseText from filePages, calls LLM, writes to documentEntities
    and entityIndex collections.

    Returns the extracted entity dict, or {} on failure.
    """
    from src.utils.connection_utils import db
    from src.utils.llm_utils import invoke_with_costing_evalution

    # Load page text
    record = db['filePages'].find_one({'fileId': file_id}, {'pageWiseText': 1, 'fileName': 1})
    if not record:
        logger.warning(f"[EntityExtractor] No filePages for fileId={file_id}")
        return {}

    raw = record.get('pageWiseText', {})
    page_wise_text = json.loads(raw) if isinstance(raw, str) else raw
    if not page_wise_text:
        return {}

    file_name = record.get('fileName', file_id)
    excerpt = _get_excerpt(page_wise_text)

    prompt = _ENTITY_EXTRACTION_PROMPT.format(document_excerpt=excerpt)

    try:
        response = invoke_with_costing_evalution(prompt)
        content = response.content.strip()

        # Strip markdown fences
        content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s*```$', '', content)

        # Extract JSON object
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

        entities = json.loads(content)

    except json.JSONDecodeError as e:
        logger.warning(f"[EntityExtractor] JSON parse failed for {file_name}: {e}")
        entities = {}
    except Exception as e:
        logger.warning(f"[EntityExtractor] LLM call failed for {file_name}: {e}")
        return {}

    # Ensure all expected keys exist
    for key in ('persons', 'organizations', 'financial_terms', 'dates', 'references', 'identifiers'):
        entities.setdefault(key, [])
    entities.setdefault('functionalRole', 'standalone')

    # Write to documentEntities (upsert)
    db['documentEntities'].update_one(
        {'fileId': file_id},
        {'$set': {
            'fileId': file_id,
            'conversationId': conversation_id,
            'fileName': file_name,
            **entities,
        }},
        upsert=True,
    )

    # Rebuild entityIndex entries for this file
    db['entityIndex'].delete_many({'fileId': file_id})

    index_records = []

    for person in entities.get('persons', []):
        name = (person.get('name') or '').strip()
        if name:
            index_records.append({
                'conversationId': conversation_id,
                'fileId': file_id,
                'entityType': 'person',
                'entityValue': name,
                'entityKey': _normalize(name),
                'entityMeta': person.get('role', ''),
            })

    for org in entities.get('organizations', []):
        name = (org.get('name') or '').strip()
        if name:
            index_records.append({
                'conversationId': conversation_id,
                'fileId': file_id,
                'entityType': 'organization',
                'entityValue': name,
                'entityKey': _normalize(name),
                'entityMeta': org.get('role', ''),
            })

    for ref in entities.get('references', []):
        value = (ref.get('value') or '').strip()
        if value:
            index_records.append({
                'conversationId': conversation_id,
                'fileId': file_id,
                'entityType': 'agreement_ref',
                'entityValue': value,
                'entityKey': _normalize(value),
                'entityMeta': ref.get('key', ''),
            })

    for ident in entities.get('identifiers', []):
        value = (ident.get('value') or '').strip()
        if value:
            index_records.append({
                'conversationId': conversation_id,
                'fileId': file_id,
                'entityType': 'project_code',
                'entityValue': value,
                'entityKey': _normalize(value),
                'entityMeta': ident.get('key', ''),
            })

    if index_records:
        db['entityIndex'].insert_many(index_records)

    # Back-propagate functionalRole to filePages so the retriever can use it
    functional_role = entities.get('functionalRole', 'standalone')
    db['filePages'].update_one(
        {'fileId': file_id},
        {'$set': {'functionalRole': functional_role}},
    )

    logger.info(
        f"[EntityExtractor] {file_name}: role={functional_role}, "
        f"{len(entities.get('persons',[]))} persons, "
        f"{len(entities.get('organizations',[]))} orgs, "
        f"{len(index_records)} index records"
    )

    return entities


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def find_files_by_entity(
    conversation_id: str,
    entity_value: str,
    entity_type: Optional[str] = None,
) -> List[str]:
    """
    Return all fileIds in the conversation that mention an entity matching
    entity_value (normalized fuzzy match on entityKey).

    entity_type filter: 'person' | 'organization' | 'agreement_ref' | 'project_code'
    Pass None to search across all types.
    """
    from src.utils.connection_utils import db

    key = _normalize(entity_value)
    if not key:
        return []

    query: dict = {
        'conversationId': conversation_id,
        'entityKey': {'$regex': re.escape(key), '$options': 'i'},
    }
    if entity_type:
        query['entityType'] = entity_type

    return list({rec['fileId'] for rec in db['entityIndex'].find(query, {'fileId': 1})})


def get_entity_summary(conversation_id: str) -> Dict[str, List[str]]:
    """
    Return a dict of {entityKey: [fileId, ...]} for every entity in the conversation.
    Useful for building the shadow graph.
    """
    from src.utils.connection_utils import db

    summary: Dict[str, List[str]] = {}
    for rec in db['entityIndex'].find(
        {'conversationId': conversation_id},
        {'entityKey': 1, 'fileId': 1, 'entityType': 1},
    ):
        key = f"{rec.get('entityType','?')}:{rec.get('entityKey','')}"
        summary.setdefault(key, [])
        fid = rec.get('fileId')
        if fid and fid not in summary[key]:
            summary[key].append(fid)

    return summary


def get_documents_for_person(conversation_id: str, person_name: str) -> List[Dict]:
    """
    Return all documents mentioning a named person, with their functionalRole
    and fileName — ordered master → transaction → modification → termination.

    Used for answering "what is Carrie Li's current contract?" without needing
    any prior classification of document types.
    """
    from src.utils.connection_utils import db

    _ROLE_ORDER = {
        'master_agreement': 0,
        'transaction': 1,
        'modification': 2,
        'termination': 3,
        'standalone': 4,
    }

    file_ids = find_files_by_entity(conversation_id, person_name, entity_type='person')
    if not file_ids:
        return []

    pages = {
        r['fileId']: r
        for r in db['filePages'].find(
            {'fileId': {'$in': file_ids}},
            {'fileId': 1, 'fileName': 1, 'functionalRole': 1, 'effectiveDate': 1, 'documentType': 1},
        )
    }

    result = []
    for fid in file_ids:
        page = pages.get(fid, {})
        role = page.get('functionalRole', 'standalone')
        result.append({
            'fileId': fid,
            'fileName': page.get('fileName', fid),
            'functionalRole': role,
            'effectiveDate': page.get('effectiveDate'),
            'documentType': page.get('documentType', ''),
            '_sort_order': _ROLE_ORDER.get(role, 99),
        })

    result.sort(key=lambda x: (x['_sort_order'], x.get('effectiveDate') or ''))
    for r in result:
        del r['_sort_order']

    return result
