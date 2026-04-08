"""
Section parser for legal contracts.

Reads pageWiseText already stored in the filePages MongoDB collection and
produces fileSections documents with normalised section IDs and real page
numbers from the PDF.

run_section_parser(file_id, file_name, conversation_id)
    Public entry point — called from processAttachmentNode after text
    extraction.  Sets section_parse_status on the filePages record and
    immediately triggers cross-reference extraction + resolution.

parse_sections(file_id, file_name, conversation_id, page_wise_text)
    Pure function — converts a pageWiseText dict into a list of section
    dicts suitable for MongoDB insertion.  Testable without any DB calls.
"""
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section detection patterns
# ---------------------------------------------------------------------------

# Numbered section at the start of a line, e.g.:
#   "12.1  Limitation of Liability"
#   "4.2(b)  The Client shall not..."
#   "1.1.1  Definitions"
#   "8.2  Payment of sums due under this Agreement shall be made to API..."
#
# Group 1: section number   Group 2: optional title / first content line
#
# NOTE: no trailing $ anchor — the line may be arbitrarily long (some PDFs
# place the full clause body on the same line as the section number).
# Title capture is capped at 250 chars; anything beyond is still accumulated
# into the section content via the _lines list.
_NUMBERED_SECTION_RE = re.compile(
    r'^(\d{1,2}(?:\.\d{1,3}){1,4}(?:\([a-z]{1,2}\))?\.?)'
    r'(?:\s{1,}(.{0,250}))?',
    re.IGNORECASE
)

# Schedule / Appendix / Exhibit header, e.g.:
#   "SCHEDULE 4"
#   "Schedule 4 – Payment and Invoice Terms"
#   "Schedule 4: Payment and Invoice Terms"   ← colon separator
#   "Appendix 1 - Service Levels"
# Group 1: full header    Group 2: optional title after separator
_SCHEDULE_RE = re.compile(
    r'^((?:SCHEDULE|Schedule|APPENDIX|Appendix|EXHIBIT|Exhibit)\s+\d+)'
    r'(?:\s*[:–\-]\s*(.+))?$'
)

# Article header: "ARTICLE I", "Article III", "Article 1"
# Group 1: full header    Group 2: optional title after dash
_ARTICLE_RE = re.compile(
    r'^((?:ARTICLE|Article)\s+(?:[IVX]+|\d+))'
    r'(?:\s*[-–—]\s*(.+))?$'
)

# Section with dotted number: "Section 2.01", "Section 1.01 – Definitions"
# Group 1: full "Section N.NN"    Group 2: optional title
_SECTION_DOTNUM_RE = re.compile(
    r'^(Section\s+\d+\.\d+(?:\.\d+)?)'
    r'(?:\s*[-–—]?\s*(.{0,200}))?$',
    re.IGNORECASE,
)

# Top-level bare section header: "11 Confidentiality", "16 Dispute Resolution"
# Requires a title (so bare "11" or list items "3 The parties agree that..." are excluded).
# Discriminators:
#   - Title must start with two letters (capital then letter — rules out "3 A." noise)
#   - No commas or periods inside the title (headings don't have them; sentences do)
#   - End-of-line anchor ($) — a sentence continues past the line, a heading ends it
#   - Title 4–60 chars total
# Group 1: section number    Group 2: title
_TOP_LEVEL_SECTION_RE = re.compile(
    r'^(\d{1,2})\.?\s+([A-Z][A-Za-z][A-Za-z\s\-&\'\/]{2,58})$'
)

# Minimum content length — sections shorter than this are likely noise
# (page headers, footers, stray numbers).
_MIN_CONTENT_CHARS = 15

# Confidence thresholds for section density (sections per page)
_MIN_SECTIONS_PER_PAGE = 0.3   # below → too few sections detected
_MAX_SECTIONS_PER_PAGE = 8.0   # above → likely over-matching body text


def _normalize_section_id(raw: str) -> str:
    """
    Canonical form for a section identifier used in MongoDB lookups.

    Numbered sections keep their case: "12.1", "4.2(b)", "1.1.1"
    Schedule/Appendix/Exhibit/Article headers are lowercased: "schedule 4", "article i"
    Section N.NN — keyword stripped, number kept: "Section 2.01" → "2.01"
    Trailing dots are stripped.
    """
    s = raw.strip().rstrip('.')
    if re.match(r'^(?:schedule|appendix|exhibit|article)', s, re.IGNORECASE):
        return s.lower()
    # "Section 2.01" → strip keyword, keep number
    m = re.match(r'^Section\s+(\d+\.\d+(?:\.\d+)?)$', s, re.IGNORECASE)
    if m:
        return m.group(1)
    return s


def _classify_line(line: str) -> Optional[Tuple[str, str]]:
    """
    If `line` is a section header, return (section_id, title_text).
    Returns None for body text.

    Detection order (most specific first):
      1. Schedule / Appendix / Exhibit
      2. Article (US-style top-level)
      3. Section N.NN (US-style dotted subsection)
      4. Decimal N.N / N.N.N (UK-style, most common)
    """
    stripped = line.strip()
    if not stripped:
        return None

    # 1. Schedule / Appendix / Exhibit
    m = _SCHEDULE_RE.match(stripped)
    if m:
        header = m.group(1).strip()
        title = (m.group(2) or '').strip()
        return _normalize_section_id(header), title

    # 2. Article header (e.g. "ARTICLE I – Definitions")
    m = _ARTICLE_RE.match(stripped)
    if m:
        header = m.group(1).strip()
        title = (m.group(2) or '').strip()
        return _normalize_section_id(header), title

    # 3. Section N.NN (e.g. "Section 2.01 – Services")
    m = _SECTION_DOTNUM_RE.match(stripped)
    if m:
        section_id = _normalize_section_id(m.group(1).strip())
        title = (m.group(2) or '').strip()
        return section_id, title

    # 4. Decimal numbered section (must have at least one dot: "N.N" minimum)
    m = _NUMBERED_SECTION_RE.match(stripped)
    if m:
        raw_id = m.group(1)
        if '.' not in raw_id.rstrip('.'):
            return None  # bare list item like "3." — skip
        section_id = _normalize_section_id(raw_id)
        title = (m.group(2) or '').strip()
        return section_id, title

    # 5. Top-level bare section with title (e.g. "11 Confidentiality")
    # Only fires when the line ends after the title (heading, not body text).
    m = _TOP_LEVEL_SECTION_RE.match(stripped)
    if m:
        section_id = m.group(1).strip()
        title = m.group(2).strip()
        return section_id, title

    return None


def parse_sections(
    file_id: str,
    file_name: str,
    conversation_id: str,
    page_wise_text: Dict[str, str],
) -> List[dict]:
    """
    Parse a pageWiseText dict into a list of section documents.

    Each document conforms to the fileSections schema:
        fileId, fileName, conversationId,
        sectionId        — normalised identifier, e.g. "12.1" or "schedule 4"
        sectionTitle     — heading text found on the header line (may be empty)
        pageNumber       — real PDF page where the section starts (int)
        content          — full section text including the header line
        parentSectionId  — e.g. "12" for section "12.1", None for top-level
    """
    # TOC page detection — a page is considered a TOC if it contains many dotted
    # leader lines like "1.  Definitions ............. 3".  We skip preamble
    # accumulation on TOC pages so TOC entries don't become the PREAMBLE chunk.
    _TOC_LINE_RE = re.compile(r'\.{4,}\s*\d+\s*$')  # "........ 14" at end of line

    def _is_toc_page(text: str) -> bool:
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return False
        toc_hits = sum(1 for l in lines if _TOC_LINE_RE.search(l))
        return toc_hits >= 3 and toc_hits / len(lines) >= 0.25

    # Preamble signature — text that indicates we're in the actual agreement opening,
    # not a cover page or TOC.  Any match → this page is a candidate preamble page.
    _PREAMBLE_SIG_RE = re.compile(
        r'\b(this\s+agreement|entered\s+into|made\s+(?:as\s+of|on)|'
        r'hereby\s+agree|witnesseth|whereas|by\s+and\s+between|'
        r'share\s+purchase|purchase\s+agreement|employment\s+agreement|'
        r'non[\-\s]disclosure)\b',
        re.IGNORECASE,
    )

    sections: List[dict] = []
    current: Optional[dict] = None  # in-progress section accumulator
    current_schedule: Optional[str] = None  # e.g. "schedule 7" once inside a Schedule
    # Lines before the first section header — captured as a PREAMBLE section.
    preamble_lines: List[str] = []
    first_section_found: bool = False
    preamble_page: int = 1
    on_preamble_page: bool = False  # True once we've passed the TOC/cover pages

    def _page_int(key: str) -> int:
        """Accept '1', 'Page-1', 'page_1', etc."""
        return int(re.sub(r'[^0-9]', '', key) or '0')

    sorted_pages = sorted(page_wise_text.items(), key=lambda x: _page_int(x[0]))

    for page_num_str, page_text in sorted_pages:
        if not page_text:
            continue
        page_num = _page_int(page_num_str)

        # Once we find a page with preamble signatures (agreement language),
        # start accumulating preamble lines.  Skip TOC/cover pages entirely.
        if not first_section_found and not on_preamble_page:
            if _PREAMBLE_SIG_RE.search(page_text) and not _is_toc_page(page_text):
                on_preamble_page = True
                preamble_page = page_num

        for line in page_text.split('\n'):
            header = _classify_line(line)
            if header is not None:
                section_id, title = header

                # Emit preamble section the first time we hit a real header
                if not first_section_found:
                    first_section_found = True
                    preamble_content = '\n'.join(preamble_lines).strip()
                    if len(preamble_content) >= _MIN_CONTENT_CHARS:
                        sections.append({
                            'fileId': file_id,
                            'fileName': file_name,
                            'conversationId': conversation_id,
                            'sectionId': 'PREAMBLE',
                            'sectionTitle': 'Preamble / Parties',
                            'pageNumber': preamble_page,
                            'content': preamble_content,
                            'parentSectionId': None,
                            '_lines': [],
                        })
                        # Remove _lines helper key right away
                        sections[-1].pop('_lines', None)

                # Finalise previous section
                if current is not None:
                    content = '\n'.join(current['_lines']).strip()
                    if len(content) >= _MIN_CONTENT_CHARS:
                        current['content'] = content
                        del current['_lines']
                        sections.append(current)

                # Track which Schedule/Appendix/Exhibit we are inside.
                # Once set it only changes when the next Schedule header appears —
                # sub-sections within a Schedule use their own numbering (1, 2, 2.1)
                # so we must not clear the context on bare integer headings.
                if re.match(r'^(?:schedule|appendix|exhibit)', section_id, re.IGNORECASE):
                    current_schedule = section_id

                # Determine parent section ID
                parent_id = _derive_parent(section_id)

                current = {
                    'fileId': file_id,
                    'fileName': file_name,
                    'conversationId': conversation_id,
                    'sectionId': section_id,
                    'sectionTitle': title,
                    'pageNumber': page_num,
                    'parentSectionId': parent_id,
                    '_lines': [line.strip()],
                }
                # Attach schedule context to every section parsed inside a Schedule.
                # The Schedule header itself gets scheduleContext too — gives the LLM
                # the full path: "Schedule 7 §schedule 7" → rendered as "Schedule 7".
                if current_schedule:
                    current['scheduleContext'] = current_schedule
            else:
                if current is not None:
                    current['_lines'].append(line.strip())
                elif not first_section_found and on_preamble_page:
                    stripped = line.strip()
                    if stripped:
                        preamble_lines.append(stripped)

    # Flush last section
    if current is not None:
        content = '\n'.join(current['_lines']).strip()
        if len(content) >= _MIN_CONTENT_CHARS:
            current['content'] = content
            del current['_lines']
            sections.append(current)

    logger.info(f"[SectionParser] Parsed {len(sections)} section(s) from {file_name}")
    return sections


def _derive_parent(section_id: str) -> Optional[str]:
    """
    "12.1"     → "12"
    "12.1.1"   → "12.1"
    "4.2(b)"   → "4.2"
    "2.01"     → "2"          (Article/Section style)
    "article i"→ None         (top-level)
    "schedule 4" → None
    """
    if re.match(r'^(?:schedule|appendix|exhibit|article)', section_id, re.IGNORECASE):
        return None
    # Trailing sub-item like "(b)" — parent is the base without the sub-item
    # e.g. "4.2(b)" → "4.2"  (not "4")
    m = re.search(r'\([a-z]{1,2}\)$', section_id)
    if m:
        return section_id[:m.start()]
    parts = section_id.split('.')
    if len(parts) > 1:
        return '.'.join(parts[:-1])
    return None


def _extract_inline_subsections(sections: List[dict]) -> List[dict]:
    """
    Option 3 — Post-parse inline sub-clause extraction.

    Some PDFs embed sub-clauses inline within a parent section's body text
    rather than starting them on separate lines, e.g. section 12 contains
    "12.1  text... 12.2  text..." without line-breaks between them.
    The main regex parser misses these because _classify_line only sees
    individual lines.

    For each stored section whose sectionId is a bare integer (e.g. "12"),
    this function scans the content for sub-clause markers (e.g. "12.1",
    "12.2") and extracts them as new child sections using the *next marker*
    as the end boundary.

    Only adds children not already present in the existing section set.
    Returns the original list extended with any new child sections.
    """
    existing_ids = {s['sectionId'] for s in sections}
    new_sections: List[dict] = []

    for sec in sections:
        sid = sec.get('sectionId', '')
        content = sec.get('content', '')

        # Only attempt extraction on bare-integer top-level sections
        # (e.g. "12", "15") with enough content to contain sub-clauses.
        if not re.match(r'^\d{1,2}$', sid) or len(content) < 50:
            continue

        # Match "{sid}.{n}" sub-clause markers that are either:
        #   • at the start of a line (possibly indented)
        #   • preceded by 2+ spaces (inline continuation after another clause)
        # Followed by whitespace + a word character (the clause body begins).
        child_re = re.compile(
            r'(?:(?:^|(?<=\n))[ \t]*|(?<=  ))'
            r'(' + re.escape(sid) + r'\.\d{1,3}(?:\.\d{1,3})?)'
            r'(?=[\s.:–\-])',
        )

        matches = list(child_re.finditer(content))
        if not matches:
            continue

        for i, m in enumerate(matches):
            child_id = m.group(1).rstrip('.')
            if child_id in existing_ids:
                continue

            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            child_content = content[start:end].strip()

            if len(child_content) < _MIN_CONTENT_CHARS:
                continue

            new_sections.append({
                'fileId': sec['fileId'],
                'fileName': sec['fileName'],
                'conversationId': sec['conversationId'],
                'sectionId': child_id,
                'sectionTitle': '',
                'pageNumber': sec['pageNumber'],
                'parentSectionId': sid,
                'content': child_content,
            })
            existing_ids.add(child_id)

    if new_sections:
        logger.info(
            f"[SectionParser] Extracted {len(new_sections)} inline sub-section(s)"
        )

    return sections + new_sections


def _deduplicate_sections(sections: List[dict]) -> List[dict]:
    """
    Remove Table-of-Contents shadow entries.

    Legal contracts have a TOC where every section number appears as a short
    line, e.g. "12.1  Limitation of Liability.....14".  These match the
    section-header regex and get stored alongside the real clause body.
    When two or more entries share the same sectionId we keep only the one
    with the longest content — that is always the real clause body.

    Insertion order (page order) is preserved for the surviving entry.
    """
    seen: Dict[str, dict] = {}
    order: List[str] = []  # tracks first-seen insertion order

    for sec in sections:
        sid = sec['sectionId']
        if sid not in seen:
            seen[sid] = sec
            order.append(sid)
        else:
            # Replace only if this copy has more content
            if len(sec.get('content', '')) > len(seen[sid].get('content', '')):
                seen[sid] = sec

    return [seen[sid] for sid in order]


def _compute_confidence(sections: list, page_wise_text: dict) -> float:
    """
    Estimate parse quality as a float in [0.0, 1.0].

    Based on section density (sections per page):
      - 0 sections found                    → 0.0
      - < 0.3 sections/page                 → 0.3  (missed most headers)
      - > 8.0 sections/page                 → 0.4  (over-matched body text)
      - otherwise                           → 0.9

    The structured expansion in full_document_loader only activates when
    confidence >= 0.7, so a score of 0.9 enables it and 0.3/0.4 disables it.
    """
    if not sections:
        return 0.0
    page_count = len(page_wise_text)
    ratio = len(sections) / max(page_count, 1)
    if ratio < _MIN_SECTIONS_PER_PAGE:
        return 0.3
    if ratio > _MAX_SECTIONS_PER_PAGE:
        return 0.4
    return 0.9


# ---------------------------------------------------------------------------
# Clause type classifier (one batch LLM call per document at index time)
# ---------------------------------------------------------------------------

_CLAUSE_TYPES = [
    "definitions",
    "order_of_precedence",
    "term_and_termination",
    "limitation_of_liability",
    "confidentiality",
    "intellectual_property",
    "governing_law",
    "dispute_resolution",
    "payment_terms",
    "data_protection",
    "warranties",
    "indemnification",
    "force_majeure",
    "notices",
    "schedule_or_appendix",
    "business_continuity",
    "audit_rights",
    "subcontracting",
    "general",
    "other",
]

# Hedging / qualifier language that limits or conditions a commitment.
# Sections matching this are flagged qualifierFlag=True at index time so
# Phase 2 retrieval can inject them alongside the commitment they qualify.
_QUALIFIER_RE = re.compile(
    r'\b(?:'
    r'no(?:t)?\s+warrant[sy]|makes?\s+no\s+warrant[sy]|'
    r'cannot\s+guarantee|does?\s+not\s+guarantee|no\s+guarantee|'
    r'best\s+endeavou?rs|reasonable\s+endeavou?rs|reasonable\s+steps|'
    r'notwithstanding|subject\s+to\s+clause|'
    r'limitation\s+of\s+liability|limit(?:s|ed)?\s+(?:its\s+)?liability|'
    r'shall\s+not\s+(?:be\s+)?liable|exclud(?:e|es|ing)\s+(?:any\s+)?liability|'
    r'exclusive\s+remedy|sole\s+remedy|'
    r'makes?\s+no\s+warranty|provided\s+(?:always\s+)?that|'
    r'except\s+(?:as|where|to\s+the\s+extent)'
    r')\b',
    re.IGNORECASE,
)

# Matches defined terms of the form:  "Term" means ...
# Pattern A — quoted:  handles straight, smart/curly, German low-9, angle, backtick,
#             and single-quote variants commonly produced by PDF text extraction.
# Pattern B — unquoted: some PDF extractors drop the quotes entirely, leaving
#             just  "Term means ..."  where Term starts with a capital letter and
#             is followed by the keyword "means" after whitespace.
_QUOTE_CHARS = r'[\u201c\u201d\u2018\u2019\u0022\u0060\u201e\u00ab\u00bb\u2039\u203a]'
_DEFINED_TERM_RE = re.compile(
    r'(?:'
    # Pattern A — quoted term
    + _QUOTE_CHARS + r'([A-Z][A-Za-z\s\-]{1,60}?)' + _QUOTE_CHARS + r'|'
    # Pattern B — unquoted capital-led term (2+ words, or single word 4+ chars)
    r'(?<!\w)((?:[A-Z][a-z]{0,30}\s+){1,5}[A-Z][a-z]{0,30}|[A-Z][A-Z]{3,30})'
    r')'
    r'\s+means\s+',
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Document-level intelligence (date, type, rank)
# ---------------------------------------------------------------------------

# Patterns for effective / execution date extraction.
# Looks for dates in "Dated 30 March 2023", "effective 1 January 2024",
# "entered into as of 15 Oct 2022", or bare "11 October 2021" near the top.
_DATE_KEYWORD_RE = re.compile(
    r'(?:dated?|effective(?:\s+as\s+of)?|entered\s+into(?:\s+as\s+of)?|made\s+(?:the\s+)?|as\s+of'
    r'|issue\s+date\s*[:=]?|issued\s*[:=]?|execution\s+date\s*[:=]?|signed\s*[:=]?)\s*'
    r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
    r'|\d{4}-\d{2}-\d{2}'
    r'|\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})',
    re.IGNORECASE,
)
_BARE_DATE_RE = re.compile(
    r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
    re.IGNORECASE,
)

# Document type keywords mapped to (documentType, documentRank)
# Rank: 0 = Master/Framework (most general), 1 = Amendment/Novation,
#       2 = Addendum/Work Order/SOW (engagement-specific), 3 = Termination/Notice
_DOC_TYPE_PATTERNS = [
    (re.compile(r'\b(termination|notice\s+of\s+termination)\b', re.IGNORECASE),  'termination_notice', 3),
    (re.compile(r'\b(work\s+order|statement\s+of\s+work|sow)\b',  re.IGNORECASE), 'work_order',         2),
    (re.compile(r'\b(addendum|side\s+letter|letter\s+of\s+variation)\b', re.IGNORECASE), 'addendum',    1),
    (re.compile(r'\b(novation|assignment|transfer)\b',             re.IGNORECASE), 'novation',           1),
    (re.compile(r'\b(amendment|variation|supplement|restatement)\b', re.IGNORECASE), 'amendment',       1),
    (re.compile(r'\b(master|framework|principal)\b',               re.IGNORECASE), 'framework',          0),
]

# Functional role — schema-agnostic legal purpose, regardless of company naming convention.
# This is the fast regex pre-fill; entity_extractor.py overwrites it with LLM precision.
# Roles: master_agreement | transaction | modification | termination | standalone
_FUNCTIONAL_ROLE_PATTERNS = [
    # termination beats everything — most specific signal
    (re.compile(r'\b(termination\s+(?:letter|notice)|notice\s+of\s+termination)\b', re.IGNORECASE), 'termination'),
    # transaction = any individual engagement under a master
    (re.compile(r'\b(work\s+order|statement\s+of\s+work|purchase\s+order|task\s+order|sow)\b', re.IGNORECASE), 'transaction'),
    # modification = changes or transfers an existing agreement
    (re.compile(r'\b(novation|amendment|side\s+letter|addendum|variation|supplement|restatement)\b', re.IGNORECASE), 'modification'),
    # master = root hub contract
    (re.compile(r'\b(master\s+(?:consultancy|services|framework)|framework\s+agreement|principal\s+agreement)\b', re.IGNORECASE), 'master_agreement'),
]


def _classify_functional_role(file_name: str, page_wise_text: Dict[str, str]) -> str:
    """
    Fast regex-based functional role assignment.  Superseded by the LLM entity
    extractor once that runs — but provides an immediate value at parse time.

    Returns one of: master_agreement | transaction | modification | termination | standalone
    """
    def _page_int(key: str) -> int:
        return int(re.sub(r'[^0-9]', '', key) or '0')

    sorted_pages = sorted(page_wise_text.items(), key=lambda x: _page_int(x[0]))
    first_page = sorted_pages[0][1] if sorted_pages else ''
    search_text = file_name + '\n' + first_page[:1500]

    for pattern, role in _FUNCTIONAL_ROLE_PATTERNS:
        if pattern.search(search_text):
            return role

    return 'standalone'


def _extract_document_date(page_wise_text: Dict[str, str]) -> Optional[str]:
    """
    Extract the effective / execution date from the first two pages of the document.

    Returns an ISO-format string (YYYY-MM-DD) on success, None if no date is found.
    Date strings are normalised via Python's datetime parser.
    """
    import calendar

    _MONTH_MAP = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

    def _parse(raw: str) -> Optional[str]:
        raw = raw.strip().rstrip('.')
        # Try ISO first
        m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', raw)
        if m:
            return raw
        # "DD Month YYYY" (with optional ordinal suffix)
        m = re.match(
            r'^(\d{1,2})(?:st|nd|rd|th)?\s+'
            r'(January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+(\d{4})$',
            raw, re.IGNORECASE,
        )
        if m:
            day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            month = _MONTH_MAP.get(month_name)
            if month:
                return f"{year:04d}-{month:02d}-{day:02d}"
        # Slash / hyphen / dot formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        m = re.match(r'^(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{4})$', raw)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return None

    def _page_int(key: str) -> int:
        return int(re.sub(r'[^0-9]', '', key) or '0')

    sorted_pages = sorted(page_wise_text.items(), key=lambda x: _page_int(x[0]))
    search_text = '\n'.join(text for _, text in sorted_pages[:2])

    # Prefer keyword-anchored dates first (more reliable)
    for match in _DATE_KEYWORD_RE.finditer(search_text):
        parsed = _parse(match.group(1))
        if parsed:
            return parsed

    # Fall back to first bare date found
    for match in _BARE_DATE_RE.finditer(search_text):
        parsed = _parse(match.group(1))
        if parsed:
            return parsed

    return None


def _classify_document_type(file_name: str, page_wise_text: Dict[str, str]) -> Tuple[str, int]:
    """
    Infer the document type and hierarchical rank from the file name and
    the first-page text.

    Returns (documentType, documentRank) where rank:
      0 = Master / Framework Agreement  (most general)
      1 = Amendment / Novation / Side Letter
      2 = Work Order / SOW / Addendum  (engagement-specific)
      3 = Termination Notice / Expiry Notice

    Lower rank = more general; higher rank = more specific.
    """
    def _page_int(key: str) -> int:
        return int(re.sub(r'[^0-9]', '', key) or '0')

    sorted_pages = sorted(page_wise_text.items(), key=lambda x: _page_int(x[0]))
    first_page = sorted_pages[0][1] if sorted_pages else ''
    search_text = file_name + '\n' + first_page[:1500]

    for pattern, doc_type, rank in _DOC_TYPE_PATTERNS:
        if pattern.search(search_text):
            return doc_type, rank

    return 'agreement', 1  # safe default — treat unknown as addendum-level


# ---------------------------------------------------------------------------
# Clause type + risk classification (one batch LLM call per document)
# ---------------------------------------------------------------------------

_RISK_LEVELS = {1, 2, 3, 4, 5}

# Risk level criteria sent to the LLM:
_RISK_GUIDANCE = """
Risk level guide (1–5):
  1 — Administrative / boilerplate (definitions, notices, counterparts, governing law).
  2 — Standard mutual obligation; industry-normal; balanced.
  3 — Slightly one-sided but within market practice (e.g. 12-month liability cap).
  4 — Above market / aggressive: asymmetric liability, broad indemnity, IP assignment
      without carve-outs, long auto-renewal lock-ins, uncapped client obligations.
  5 — Red Flag: unlimited indemnity on one side only, IP vested in supplier for client
      data, no cap on termination charges, "sole discretion" fee changes, "time of the
      essence" on client payments with immediate termination right.
"""


def _classify_clause_types(sections: List[dict]) -> List[dict]:
    """
    Classify each section by clause type AND risk level.

    Two-stage hybrid:
      Stage 1 — Fine-tuned BERT classifier (clause_classifier_instance) assigns
                 clauseType for every section.  Fast, free, runs locally.
                 Falls back gracefully if the model directory does not exist yet
                 (pre-training) — in that case Stage 2 handles clauseType too.

      Stage 2 — Single LLM call assigns riskLevel + riskNote for all sections.
                 When Stage 1 ran successfully the LLM prompt omits clauseType,
                 making the response ~60% smaller and faster.
                 When Stage 1 was not available the LLM also assigns clauseType
                 (original behaviour, unchanged).

    Non-critical — if both stages fail, sections are returned unchanged.
    Cost (post-training): one small LLM call per document for risk only.
    Cost (pre-training):  one LLM call per document for clause type + risk
                          (identical to previous behaviour).
    """
    if not sections:
        return sections

    from src.utils.llm_utils import invoke_with_costing_evalution
    from src.utils.clause_classifier_instance import (
        classify_sections_batch,
        is_available as classifier_available,
    )

    # ── Stage 1: BERT clause type classification ──────────────────────────────
    bert_ran = classifier_available()
    if bert_ran:
        classify_sections_batch(sections)
        logger.info(
            "[SectionParser] Stage 1: BERT classifier assigned clauseType "
            "for %d section(s).", len(sections)
        )

    # ── Stage 2: LLM risk level + (optionally) clause type ───────────────────
    # Skip entirely when running in backfill training-data mode
    if os.environ.get("BACKFILL_SKIP_RISK"):
        logger.info("[SectionParser] Stage 2 skipped (BACKFILL_SKIP_RISK mode)")
        return sections

    lines = []
    for s in sections:
        preview = s.get("content", "")[:200].replace("\n", " ")
        title = s.get("sectionTitle", "")
        sid = s.get("sectionId", "")
        lines.append(f"{sid} | {title} | {preview}")

    section_text = "\n".join(lines)

    if bert_ran:
        # Narrower prompt — only risk assessment needed
        prompt = (
            f"You are a senior contract analyst. Assign a risk level to each section.\n\n"
            f"{_RISK_GUIDANCE}\n"
            f"Each line is: sectionId | sectionTitle | content preview\n\n"
            f"{section_text}\n\n"
            f"Return ONLY a JSON array. Each element must have:\n"
            f"  sectionId  — exactly as given\n"
            f"  riskLevel  — integer 1–5 per the guide above\n"
            f"  riskNote   — one sentence describing the specific risk if riskLevel >= 4, "
            f"otherwise \"\"\n\n"
            f"Example: [{{\"sectionId\": \"12.1\", \"riskLevel\": 4, "
            f"\"riskNote\": \"Liability cap applies only to supplier, not client.\"}}]"
        )
    else:
        # Full prompt — BERT not available, LLM handles clauseType as well
        clause_type_list = ", ".join(_CLAUSE_TYPES)
        prompt = (
            f"You are a senior contract analyst. Classify each contract section below by clause type "
            f"and assign a risk level.\n\n"
            f"Allowed clause types: {clause_type_list}\n\n"
            f"Key guidance on clause types:\n"
            f"  order_of_precedence — clauses defining which document or provision wins in a conflict "
            f"(e.g. 'takes precedence', 'shall prevail', 'order of priority', 'in the event of conflict').\n"
            f"  definitions — clauses that define terms used in the agreement.\n"
            f"  general — structural/boilerplate clauses that don't fit any specific type.\n\n"
            f"{_RISK_GUIDANCE}\n"
            f"Each line is: sectionId | sectionTitle | content preview\n\n"
            f"{section_text}\n\n"
            f"Return ONLY a JSON array. Each element must have:\n"
            f"  sectionId  — exactly as given\n"
            f"  clauseType — one value from the allowed list\n"
            f"  riskLevel  — integer 1–5 per the guide above\n"
            f"  riskNote   — one sentence describing the specific risk if riskLevel >= 4, otherwise \"\"\n\n"
            f"Example: [{{\"sectionId\": \"12.1\", \"clauseType\": \"limitation_of_liability\", "
            f"\"riskLevel\": 4, \"riskNote\": \"Liability cap applies only to supplier, not client.\"}}]"
        )

    try:
        response = invoke_with_costing_evalution(prompt=prompt)
        raw = response.content.strip()
        if "```" in raw:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            raw = raw[start:end] if start != -1 and end > start else raw
        classifications = json.loads(raw)
        if not isinstance(classifications, list):
            return sections

        class_map = {
            str(c.get("sectionId", "")).strip(): c
            for c in classifications
            if c.get("sectionId")
        }

        risk_flags = 0
        for sec in sections:
            entry = class_map.get(str(sec.get("sectionId", "")), {})

            # clauseType — only apply from LLM when BERT did not run
            if not bert_ran:
                clause_type = str(entry.get("clauseType", "other")).strip()
                if clause_type not in _CLAUSE_TYPES:
                    clause_type = "other"
                sec["clauseType"] = clause_type

            raw_risk = entry.get("riskLevel", 2)
            risk_level = int(raw_risk) if str(raw_risk).isdigit() else 2
            risk_level = max(1, min(5, risk_level))
            sec["riskLevel"] = risk_level
            risk_note = str(entry.get("riskNote", "")).strip()
            if risk_note:
                sec["riskNote"] = risk_note
            if risk_level >= 4:
                risk_flags += 1

        stage_label = "risk-only LLM" if bert_ran else "full LLM (BERT not available)"
        logger.info(
            "[SectionParser] Stage 2 (%s): %d section(s) risk-scored; "
            "%d high-risk section(s) flagged.",
            stage_label, len(class_map), risk_flags,
        )

    except Exception as e:
        logger.warning("[SectionParser] Stage 2 LLM risk classification failed: %s", e)

    return sections


def _extract_defined_terms(sections: List[dict], file_id: str) -> None:
    """
    Extract defined terms from sections classified as 'definitions' and
    store them in filePages.definedTerms as {term: snippet}.

    Called after _classify_clause_types so clauseType is already set.
    Non-critical — fails silently.
    """
    from src.utils.db_utils import upsertCollection

    def_sections = [s for s in sections if s.get("clauseType") == "definitions"]
    if not def_sections:
        return

    full_text = "\n".join(s.get("content", "") for s in def_sections)
    matches = list(_DEFINED_TERM_RE.finditer(full_text))
    if not matches:
        return

    defined_terms: dict = {}
    for i, match in enumerate(matches):
        term = match.group(1).strip()
        def_start = match.end()
        # Use start of next term as boundary; cap at 400 chars
        def_end = matches[i + 1].start() if i + 1 < len(matches) else def_start + 400
        def_end = min(def_end, def_start + 400)
        snippet = full_text[def_start:def_end].strip().replace("\n", " ")
        # Trim to last complete sentence if snippet is long
        if len(snippet) > 350 and ". " in snippet:
            snippet = snippet[: snippet.rfind(". ", 0, 350) + 1]
        if term and snippet and len(term) > 1:
            defined_terms[term] = snippet

    if not defined_terms:
        return

    try:
        upsertCollection("filePages", "fileId", file_id, {"definedTerms": defined_terms})
        logger.info(
            f"[SectionParser] Extracted {len(defined_terms)} defined term(s) for fileId={file_id}"
        )
    except Exception as e:
        logger.warning(f"[SectionParser] Failed to store defined terms: {e}")


def _flag_qualifier_sections(sections: List[dict]) -> List[dict]:
    """
    Scan each section's content for hedging / limitation language and set
    qualifierFlag=True on matches.  Pure regex — no LLM call, no cost.

    qualifierFlag is used by Phase 2 retrieval to inject these sections
    alongside whichever commitment section they qualify, ensuring the LLM
    always sees the carve-out next to the metric it limits.
    """
    flagged = 0
    for sec in sections:
        content = sec.get("content", "")
        if _QUALIFIER_RE.search(content):
            sec["qualifierFlag"] = True
            flagged += 1
    if flagged:
        logger.info(f"[SectionParser] Flagged {flagged} qualifier section(s)")
    return sections


# ---------------------------------------------------------------------------
# LLM fallback parser (only fires when regex confidence < 0.7)
# ---------------------------------------------------------------------------

def _llm_fallback_parse(
    file_id: str,
    file_name: str,
    conversation_id: str,
    page_wise_text: Dict[str, str],
) -> List[dict]:
    """
    LLM-based section detection for messy PDFs where regex confidence is low.

    Cost-efficient approach:
      1. Extract only candidate lines (lines that look like potential headers)
         — typically 50-150 lines, ~1,500 input tokens per document.
      2. LLM identifies which candidates are real section headers and returns
         structured JSON: [{sectionId, sectionTitle, pageNumber}].
      3. Re-parse the full document using those anchors to extract content
         — content extraction is still done locally, no extra LLM calls.
    """
    from src.utils.llm_utils import invoke_with_costing_evalution

    def _page_int(key: str) -> int:
        return int(re.sub(r'[^0-9]', '', key) or '0')

    sorted_pages = sorted(page_wise_text.items(), key=lambda x: _page_int(x[0]))

    # Step 1: Extract candidate lines — only lines that might be headers
    candidates = []
    for page_key, text in sorted_pages:
        if not text:
            continue
        page_num = _page_int(page_key)
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped or len(stripped) > 150:
                continue
            if re.match(
                r'^\d|^[A-Z]{2,}|^(?:SCHEDULE|APPENDIX|EXHIBIT|ARTICLE|PART|SECTION)',
                stripped,
            ) or (
                len(stripped) <= 80
                and re.match(r'^[A-Z][a-z]', stripped)
                and '.' not in stripped
            ):
                candidates.append((page_num, stripped))

    if not candidates:
        logger.warning(f"[SectionParser] LLM fallback: no candidate lines in {file_name}")
        return []

    # Cap to 150 candidates to stay within token budget
    if len(candidates) > 150:
        candidates = candidates[:150]

    candidate_text = '\n'.join(f"p{pnum}: {line}" for pnum, line in candidates)

    prompt = (
        f"You are analyzing a legal contract: \"{file_name}\".\n\n"
        f"Below are lines from the document that may be section headers, each prefixed "
        f"with their page number.\n\n"
        f"Identify which lines are genuine section or clause headers "
        f"(NOT table-of-contents entries, page numbers, or body text sentences).\n\n"
        f"Return ONLY a JSON array. Each element must have:\n"
        f"  sectionId   — the number or identifier exactly as it appears "
        f"(e.g. \"12.1\", \"schedule 4\", \"article i\")\n"
        f"  sectionTitle — heading title text, empty string if none\n"
        f"  pageNumber   — integer\n\n"
        f"Example: [{{\"sectionId\": \"12.1\", \"sectionTitle\": \"Limitation of Liability\", "
        f"\"pageNumber\": 14}}]\n\n"
        f"Candidate lines:\n{candidate_text}"
    )

    try:
        response = invoke_with_costing_evalution(prompt=prompt)
        raw = response.content.strip()
        if '```' in raw:
            start = raw.find('[')
            end = raw.rfind(']') + 1
            raw = raw[start:end] if start != -1 and end > start else raw
        headers = json.loads(raw)
        if not isinstance(headers, list):
            return []
    except Exception as e:
        logger.warning(f"[SectionParser] LLM fallback response parse failed for {file_name}: {e}")
        return []

    if not headers:
        return []

    # Step 2: Build anchor map — normalised sectionId → (sectionId, title)
    anchor_map: Dict[str, Tuple[str, str]] = {}
    for h in headers:
        sid = str(h.get('sectionId', '')).strip()
        title = str(h.get('sectionTitle', '')).strip()
        if not sid:
            continue
        norm = _normalize_section_id(sid)
        anchor_map[norm] = (norm, title)

    if not anchor_map:
        return []

    # Step 3: Re-parse the document using LLM anchors for header detection
    sections: List[dict] = []
    current: Optional[dict] = None

    for page_key, text in sorted_pages:
        if not text:
            continue
        page_num = _page_int(page_key)

        for line in text.split('\n'):
            stripped = line.strip()
            matched: Optional[Tuple[str, str]] = None

            for norm_sid, (sid, title) in anchor_map.items():
                # Line must START with the section ID followed by whitespace,
                # punctuation, or end-of-string — prevents "1" matching "12.1"
                if re.match(
                    r'^' + re.escape(norm_sid) + r'(?:[\s\.\:\-\u2013\u2014]|$)',
                    stripped,
                    re.IGNORECASE,
                ):
                    matched = (sid, title)
                    break

            if matched:
                sid, title = matched
                if current is not None:
                    content = '\n'.join(current['_lines']).strip()
                    if len(content) >= _MIN_CONTENT_CHARS:
                        current['content'] = content
                        del current['_lines']
                        sections.append(current)

                current = {
                    'fileId': file_id,
                    'fileName': file_name,
                    'conversationId': conversation_id,
                    'sectionId': sid,
                    'sectionTitle': title,
                    'pageNumber': page_num,
                    'parentSectionId': _derive_parent(sid),
                    '_lines': [stripped],
                }
            else:
                if current is not None:
                    current['_lines'].append(stripped)

    if current is not None:
        content = '\n'.join(current['_lines']).strip()
        if len(content) >= _MIN_CONTENT_CHARS:
            current['content'] = content
            del current['_lines']
            sections.append(current)

    logger.info(
        f"[SectionParser] LLM fallback parsed {len(sections)} section(s) from {file_name}"
    )
    return sections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_section_parser(file_id: str, file_name: str, conversation_id: str):
    """
    Full pipeline entry point called from processAttachmentNode.

    1. Sets section_parse_status = "parsing"
    2. Reads pageWiseText from filePages (already stored)
    3. Parses sections and writes them to fileSections
    4. Sets section_parse_status = "ready" (or "failed")
    5. Triggers cross-reference extraction + resolution
    """
    from src.utils.connection_utils import db
    from src.utils.db_utils import upsertCollection
    from src.utils.log_utils import insert_logs

    try:
        upsertCollection('filePages', 'fileId', file_id, {
            'section_parse_status': 'parsing',
        })

        record = db['filePages'].find_one({'fileId': file_id})
        if not record:
            logger.warning(f"[SectionParser] No filePages record for fileId={file_id}")
            upsertCollection('filePages', 'fileId', file_id, {
                'section_parse_status': 'failed',
            })
            return

        raw = record.get('pageWiseText', {})
        page_wise_text = json.loads(raw) if isinstance(raw, str) else raw

        if not page_wise_text:
            logger.warning(f"[SectionParser] Empty pageWiseText for fileId={file_id}")
            upsertCollection('filePages', 'fileId', file_id, {
                'section_parse_status': 'failed',
            })
            return

        sections = parse_sections(file_id, file_name, conversation_id, page_wise_text)
        sections = _deduplicate_sections(sections)
        sections = _extract_inline_subsections(sections)

        confidence = _compute_confidence(sections, page_wise_text)

        # Low confidence — regex missed most headers or over-matched body text.
        # Try LLM-based header detection and keep the result only if it scores higher.
        # Must run BEFORE writing to MongoDB so the winning result is what gets stored.
        if confidence < 0.7:
            logger.info(
                f"[SectionParser] Low regex confidence ({confidence:.2f}) for {file_name}"
                f" — triggering LLM fallback"
            )
            try:
                fallback = _llm_fallback_parse(file_id, file_name, conversation_id, page_wise_text)
                if fallback:
                    fallback = _deduplicate_sections(fallback)
                    fallback = _extract_inline_subsections(fallback)
                    fallback_conf = _compute_confidence(fallback, page_wise_text)
                    if fallback_conf > confidence:
                        sections = fallback
                        confidence = fallback_conf
                        logger.info(
                            f"[SectionParser] LLM fallback accepted for {file_name} "
                            f"(confidence {confidence:.2f})"
                        )
                    else:
                        logger.info(
                            f"[SectionParser] LLM fallback did not improve confidence "
                            f"({fallback_conf:.2f} <= {confidence:.2f}), keeping regex result"
                        )
            except Exception as fb_e:
                logger.warning(
                    f"[SectionParser] LLM fallback error for {file_name}: {fb_e}"
                )

        # Diagnostic: log all stored sectionIds so resolution failures can be
        # traced against what was actually parsed from the PDF.
        stored_ids = sorted(s['sectionId'] for s in sections)
        logger.info(
            "[SectionParser] SectionIds stored for %s (%d): %s",
            file_name, len(stored_ids), ", ".join(stored_ids),
        )

        # Classify sections by clause type (one batch LLM call)
        sections = _classify_clause_types(sections)

        # Flag sections containing hedging / limitation language
        sections = _flag_qualifier_sections(sections)

        # Extract defined terms from definitions sections → stored in filePages
        _extract_defined_terms(sections, file_id)

        # Replace any previous sections for this file atomically
        db['fileSections'].delete_many({'fileId': file_id})
        if sections:
            db['fileSections'].insert_many(sections)

        insert_logs(
            message=(
                f"[SectionParser] Stored {len(sections)} section(s) for {file_name} "
                f"(confidence={confidence:.2f})"
            ),
            logType='information',
            bulkId=file_id,
        )

        # Extract effective date, document type/rank, and functional role from the raw text.
        # functionalRole is schema-agnostic (master_agreement/transaction/modification/
        # termination/standalone) — overwritten later by entity_extractor LLM call.
        effective_date = _extract_document_date(page_wise_text)
        doc_type, doc_rank = _classify_document_type(file_name, page_wise_text)
        functional_role = _classify_functional_role(file_name, page_wise_text)

        doc_meta: dict = {
            'section_parse_status': 'ready',
            'section_parse_confidence': confidence,
            'sections_count': len(sections),
            'documentType': doc_type,
            'documentRank': doc_rank,
            'functionalRole': functional_role,
        }
        if effective_date:
            doc_meta['effectiveDate'] = effective_date

        # Collect high-risk sections for a quick per-file risk summary
        high_risk = [
            {'sectionId': s['sectionId'], 'riskLevel': s.get('riskLevel', 1), 'riskNote': s.get('riskNote', '')}
            for s in sections if s.get('riskLevel', 1) >= 4
        ]
        if high_risk:
            doc_meta['riskFlags'] = high_risk

        upsertCollection('filePages', 'fileId', file_id, doc_meta)

        insert_logs(
            message=(
                f"[SectionParser] Document metadata: type={doc_type}, rank={doc_rank}, "
                f"effectiveDate={effective_date}, riskFlags={len(high_risk)}"
            ),
            logType='information',
            bulkId=file_id,
        )

        # Extract cross-references from this file and resolve across conversation
        from src.services.crossref_extractor import (
            extract_cross_references,
            resolve_cross_references,
        )
        extract_cross_references(file_id, conversation_id)
        resolve_cross_references(conversation_id)

    except Exception as e:
        logger.error(f"[SectionParser] Failed for fileId={file_id}: {e}")
        try:
            from src.utils.db_utils import upsertCollection as _upsert
            _upsert('filePages', 'fileId', file_id, {'section_parse_status': 'failed'})
        except Exception:
            pass
