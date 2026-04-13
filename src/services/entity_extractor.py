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
- master_agreement: Root/hub contract that other documents sit under (MSA, Framework Agreement, Master Consultancy Agreement, MCA, Master Services Agreement). It does NOT reference another agreement as its parent.
- transaction: Individual engagement issued under a master (Work Order, SOW, Statement of Work, Purchase Order, Task Order, Scoping Agreement). Usually references a master agreement as its parent.
- modification: Any document that explicitly states it is an Addendum, Amendment, Novation, Supplement, Variation, or Side Letter to another agreement — even if the document itself is large or complex (e.g. a SaaS Agreement that says "this Agreement is an Addendum to the Framework Agreement"). KEY SIGNAL: phrases like "an Addendum to", "pursuant to the Framework Agreement", "incorporating the terms of", "issued under" indicate modification.
- termination: Ends an agreement (Termination Letter, Notice of Termination, Expiry Notice)
- standalone: Not linked to a hub (NDA, Policy, Standalone License, independent contract with no parent reference)

CRITICAL RULE — Addendum detection: If the document excerpt contains any of these phrases, classify as modification REGARDLESS of the document's own title:
  • "is an Addendum to"
  • "as an Addendum"
  • "addendum to the [any] Agreement"
  • "pursuant to the Framework Agreement"
  • "incorporates the terms of the [Master/Framework] Agreement"

Rules:
- Extract ALL persons mentioned (consultants, signatories, representatives)
- Extract ALL organizations (use full legal names where visible)
- financial_terms: include day rates, monthly fees, annual fees, formulas, caps, penalties. Pay special attention to any [FINANCIAL TERMS IN LATER PAGES] section appended below — fee tables in Schedules often contain the only explicit currency symbol in the document
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

_CURRENCY_RE = re.compile(
    r'(?:'
    # ISO 4217 codes — word boundaries prevent matching inside words (e.g. "necessary", "chandrasekhar")
    r'\b(?:'
    r'HKD|SGD|USD|GBP|EUR|AUD|JPY|CNY|MYR|THB|NZD|CAD|CHF'   # core
    r'|INR|IDR|KRW|PHP|TWD|BRL|ZAR|AED|SAR|QAR'               # emerging / Middle East
    r'|DKK|SEK|NOK|CZK|PLN|HUF'                                # European non-Euro
    r')\b'
    # Prefixed regional symbols (must precede digits to avoid false positives)
    r'|HK\$|S\$|US\$|A\$|NZ\$'
    r'|RM\s*\d'                                                 # Malaysian Ringgit: RM 5,000
    # Written-out names — only capture when followed by a digit or another currency word
    r'|(?:Hong\s+Kong|Singapore|US|Australian|New\s+Zealand)\s+Dollar'
    r'|(?:Pound\s+Sterling|Sterling\s+Pound|British\s+Pound)'   # GBP written out
    r'|Indian\s+Rupee|Indonesian\s+Rupiah|Philippine\s+Peso'
    r'|(?:Euro|Euros)\s*\d'                                     # "Euro 500" / "Euros 100"
    # Currency symbol immediately before a digit
    r'|[\$£€¥₹]\s*\d'
    r')',
    re.IGNORECASE,
)


def _get_excerpt(page_wise_text: Dict[str, str], max_chars: int = 2000) -> str:
    """
    Build an excerpt for entity extraction.

    Always includes the first 3 pages (parties, role, key dates).
    Additionally scans all remaining pages for currency/financial content and
    appends short snippets around any matches — so fee tables buried in
    Schedules are visible to the LLM even when they appear on page 10+.
    """
    def _page_int(key: str) -> int:
        return int(re.sub(r'[^0-9]', '', key) or '0')

    sorted_pages = sorted(page_wise_text.items(), key=lambda x: _page_int(x[0]))

    # Core: first 3 pages
    head_text = '\n'.join(text for _, text in sorted_pages[:3])

    # Financial scan: remaining pages
    currency_snippets = []
    for page_key, page_text in sorted_pages[3:]:
        m = _CURRENCY_RE.search(page_text)
        if not m:
            continue
        # Extract ~150 chars of context around the first currency mention
        start = max(0, m.start() - 120)
        end = min(len(page_text), m.end() + 200)
        snippet = f"[Page {page_key}] ...{page_text[start:end].strip()}..."
        currency_snippets.append(snippet)
        if len(currency_snippets) >= 4:   # cap — avoid ballooning the prompt
            break

    combined = head_text[:max_chars]
    if currency_snippets:
        combined += '\n\n[FINANCIAL TERMS IN LATER PAGES]\n' + '\n'.join(currency_snippets)

    return combined


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

    # Post-processing: Addendum detection safety net.
    # If the LLM classified a document as master_agreement or standalone but the
    # excerpt contains explicit addendum/incorporation language, downgrade to modification.
    # This catches SaaS Agreements, Service Schedules, and similar documents that are
    # technically Addenda to a Framework Agreement despite being large and complex.
    _ADDENDUM_RE = re.compile(
        r'\b(is\s+an\s+addendum\s+to|as\s+an\s+addendum|addendum\s+to\s+the|'
        r'pursuant\s+to\s+the\s+(?:framework|master)|'
        r'incorporates?\s+(?:the\s+)?(?:terms\s+(?:and\s+conditions\s+)?of\s+)?'
        r'the\s+(?:framework|master)|'
        r'issued\s+under\s+the\s+(?:framework|master))\b',
        re.IGNORECASE,
    )
    current_role = entities.get('functionalRole', 'standalone')
    if current_role in ('master_agreement', 'standalone') and _ADDENDUM_RE.search(excerpt):
        entities['functionalRole'] = 'modification'
        logger.info(
            "[EntityExtractor] %s: role corrected master_agreement→modification "
            "(addendum language detected in excerpt)",
            file_name,
        )

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
