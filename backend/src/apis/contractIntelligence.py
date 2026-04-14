"""
Contract Intelligence — standalone RAG endpoints.



Endpoints:
    POST /ci/upload          — receive file, store locally, parse + embed (background)
    GET  /ci/status/{fileId} — check processing status
    POST /ci/query           — full RAG pipeline, returns answer + citations
    GET  /ci/file/{fileId}   — serve raw file for PDF viewer

Processing chain :
    loadDocumentTextPageWise → filePages.pageWiseText
    run_section_parser       → fileSections + fileCrossRefs
    embed_sections_to_qdrant → Qdrant contract_sections
"""

import json
import logging
import os
import re
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.utils.connection_utils import db
from src.utils.db_utils import upsertCollection
from src.utils.log_utils import insert_logs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ci", tags=["Contract Intelligence"])

# Root directory for all locally stored CI files
_CI_FILES_ROOT = os.path.join(os.getcwd(), "ci_files")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_dir(conversation_id: str) -> str:
    path = os.path.join(_CI_FILES_ROOT, conversation_id)
    os.makedirs(path, exist_ok=True)
    return path


def _get_local_path(file_id: str) -> Optional[str]:
    """Resolve local file path from filePages record."""
    record = db["filePages"].find_one({"fileId": file_id})
    if not record:
        return None
    return record.get("localPath")


# ---------------------------------------------------------------------------
# Background processing — identical to processAttachmentNode pipeline
# ---------------------------------------------------------------------------

def _process_file(file_id: str, file_name: str, conversation_id: str, local_path: str):
    """
    Full processing pipeline run in a background thread.

    Steps (identical to processAttachmentNode.py):
        1. loadDocumentTextPageWise  → extract text page by page
        2. upsertCollection filePages → store pageWiseText
        3. run_section_parser        → fileSections + fileCrossRefs indexes
        4. embed_sections_to_qdrant  → Qdrant vector index
    """
    try:
        insert_logs(
            message=f"[CI] Processing started: {file_name}",
            logType="information",
            bulkId=conversation_id,
        )

        # Mark as in-progress
        upsertCollection("filePages", "fileId", file_id, {
            "processing_status": "pre_processing",
        })

        # Step 1 — Extract text page-wise (same loader chain as processAttachmentNode)
        from src.utils.document_utils import loadDocumentTextPageWise
        full_text, page_wise_text = loadDocumentTextPageWise(local_path, bulkId=conversation_id)

        if not page_wise_text:
            raise ValueError(f"No text could be extracted from {file_name}")

        # Step 2 — Persist page-wise text to filePages
        upsertCollection("filePages", "fileId", file_id, {
            "pageWiseText": json.dumps(page_wise_text),
            "conversationId": conversation_id,
            "fileName": file_name,
            "localPath": local_path,
            "processing_status": "pre_processing",
        })

        insert_logs(
            message=f"[CI] Page-wise text stored for {file_name} ({len(page_wise_text)} pages)",
            logType="information",
            bulkId=conversation_id,
        )

        # Step 3 — Build structural section index (cross-reference resolution)
        from src.services.section_parser import run_section_parser
        run_section_parser(file_id, file_name, conversation_id)

        insert_logs(
            message=f"[CI] Section index built for {file_name}",
            logType="information",
            bulkId=conversation_id,
        )

        # Step 4 — Embed sections into Qdrant for vector retrieval
        from src.services.section_embedder import embed_sections_to_qdrant
        n = embed_sections_to_qdrant(file_id, conversation_id)

        insert_logs(
            message=f"[CI] Embedded {n} section(s) to Qdrant for {file_name}",
            logType="information",
            bulkId=conversation_id,
        )

        # Step 5 — Extract entity index (needed before graph extraction)
        try:
            from src.services.entity_extractor import extract_entities
            extract_entities(file_id, conversation_id)
            insert_logs(
                message=f"[CI] Entity extraction complete for {file_name}",
                logType="information",
                bulkId=conversation_id,
            )
        except Exception as ent_e:
            insert_logs(
                message=f"[CI] Entity extraction warning for {file_name}: {ent_e}",
                logType="warning",
                bulkId=conversation_id,
            )

        # Step 6 — Build document relationships (master_child, novation, termination, renewal)
        # Must run BEFORE graph extraction so MASTER_OF / NOVATES / TERMINATES edges exist.
        try:
            from src.services.document_relationship_service import build_document_relationships
            build_document_relationships(conversation_id)
            insert_logs(
                message=f"[CI] Document relationships built for conversation {conversation_id}",
                logType="information",
                bulkId=conversation_id,
            )
        except Exception as rel_e:
            insert_logs(
                message=f"[CI] Document relationship warning: {rel_e}",
                logType="warning",
                bulkId=conversation_id,
            )

        # Step 7 — Build knowledge graph (depends on sections + entities + relationships)
        try:
            from src.services.graph_extractor import extract_graph
            graph_summary = extract_graph(file_id, conversation_id)
            insert_logs(
                message=(
                    f"[CI] Graph extraction complete for {file_name}: "
                    f"{graph_summary.get('nodes', 0)} nodes, "
                    f"{graph_summary.get('edges', 0)} edges "
                    f"({graph_summary.get('llm_edges', 0)} LLM-extracted)"
                ),
                logType="information",
                bulkId=conversation_id,
            )
        except Exception as graph_e:
            insert_logs(
                message=f"[CI] Graph extraction warning for {file_name}: {graph_e}",
                logType="warning",
                bulkId=conversation_id,
            )

        # Mark ready — sections_embedded is set by embed_sections_to_qdrant only
        # after Qdrant confirms the write succeeded. Do NOT set it here or it will
        # mask a silent upsert failure and prevent any retry.
        upsertCollection("filePages", "fileId", file_id, {
            "processing_status": "ready",
        })
        upsertCollection("files", "fileId", file_id, {"status": "Processed"})

        insert_logs(
            message=f"[CI] Processing complete: {file_name}",
            logType="information",
            bulkId=conversation_id,
        )

    except Exception as e:
        logger.error("[CI] Processing failed for %s: %s", file_name, e)
        insert_logs(
            message=f"[CI] Processing failed for {file_name}: {e}",
            logType="error",
            bulkId=conversation_id,
        )
        upsertCollection("filePages", "fileId", file_id, {"processing_status": "failed"})
        upsertCollection("files", "fileId", file_id, {"status": "Failed"})


# ---------------------------------------------------------------------------
# POST /ci/upload
# ---------------------------------------------------------------------------

@router.post("/upload", summary="Upload and process a contract file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conversationId: Optional[str] = Form(None),
):
    """
    Receive a file from the frontend, save it locally, and kick off the full
    processing pipeline (parse + embed) in the background.

    Returns immediately with { fileId, fileName, conversationId } so the
    frontend can attach the file to the conversation and start querying.
    Processing status is queryable via GET /ci/status/{fileId}.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Generate IDs
    file_id = str(uuid.uuid4())
    conversation_id = conversationId or str(uuid.uuid4())

    file_name = file.filename
    dest_dir = _file_dir(conversation_id)
    local_path = os.path.join(dest_dir, file_name)

    # Save file to disk
    contents = await file.read()
    with open(local_path, "wb") as f:
        f.write(contents)

    insert_logs(
        message=f"[CI] File saved: {file_name} → {local_path}",
        logType="information",
        bulkId=conversation_id,
    )

    # Create initial MongoDB records
    upsertCollection("files", "fileId", file_id, {
        "fileId": file_id,
        "fileName": file_name,
        "conversationId": conversation_id,
        "status": "Queued",
        "blobName": "",
        "additionalFields": {"localPath": local_path},
    })

    upsertCollection("filePages", "fileId", file_id, {
        "fileId": file_id,
        "fileName": file_name,
        "conversationId": conversation_id,
        "localPath": local_path,
        "processing_status": "queued",
        "sections_embedded": False,
    })

    # Create conversation record (enables clientId scoping for playbooks later)
    db["conversations"].update_one(
        {"conversationId": conversation_id},
        {"$setOnInsert": {"conversationId": conversation_id, "clientId": None}},
        upsert=True,
    )

    # Kick off background processing (identical pipeline to processAttachmentNode)
    background_tasks.add_task(
        _process_file, file_id, file_name, conversation_id, local_path
    )

    return {
        "fileId": file_id,
        "fileName": file_name,
        "conversationId": conversation_id,
        "status": "processing",
    }


# ---------------------------------------------------------------------------
# GET /ci/status/{fileId}
# ---------------------------------------------------------------------------

@router.get("/status/{file_id}", summary="Check file processing status")
def get_status(file_id: str):
    """
    Returns processing_status for a file:
        queued       — waiting to start
        pre_processing — actively being parsed/embedded
        ready        — sections indexed, ready to query
        failed       — processing error
    """
    record = db["filePages"].find_one({"fileId": file_id}, {"processing_status": 1, "sections_embedded": 1})
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "fileId": file_id,
        "processing_status": record.get("processing_status", "unknown"),
        "sections_embedded": record.get("sections_embedded", False),
    }


# ---------------------------------------------------------------------------
# POST /ci/query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    conversationId: str
    query: str


@router.post("/query", summary="RAG query against uploaded contracts")
def query_contracts(body: QueryRequest):
    """
    Full RAG pipeline — identical to askAttachmentsNode.py.

    Pipeline:
        retrieve_sections (Phases 0-4b) → build prompt → LLM answer → parse citations

    Returns:
        { answer, citations: [{ fileId, fileName, PageNumber, TextToFind }] }
    """
    conversation_id = body.conversationId
    query = body.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    insert_logs(
        message=f"[CI] Query received for conversation {conversation_id}: {query[:100]}",
        logType="information",
        bulkId=conversation_id,
    )

    # --- Retrieve sections (full Phase 0-4b pipeline) ---
    from src.services.section_retriever import retrieve_sections, all_sections_embedded

    if all_sections_embedded(conversation_id):
        try:
            relevant_context, context_metadata = retrieve_sections(conversation_id, query)
            if not relevant_context:
                raise ValueError("Section retriever returned empty context")
        except Exception as retr_e:
            insert_logs(
                message=f"[CI] Section retriever failed, falling back to full doc load: {retr_e}",
                logType="warning",
                bulkId=conversation_id,
            )
            from src.utils.full_document_loader import load_full_documents_with_citations
            relevant_context, context_metadata = load_full_documents_with_citations(conversation_id)
    else:
        from src.utils.full_document_loader import load_full_documents_with_citations
        relevant_context, context_metadata = load_full_documents_with_citations(conversation_id)

    if not relevant_context:
        raise HTTPException(
            status_code=422,
            detail="Could not retrieve document content. Ensure files are fully processed before querying.",
        )

    # --- Build document relationship map + entity thread (identical to askAttachmentsNode) ---
    relationship_context = ""
    entity_thread_context = ""
    try:
        from src.services.document_relationship_service import build_document_relationships
        from src.services.entity_extractor import get_entity_summary, get_documents_for_person

        rels = list(db["documentRelationships"].find(
            {"conversationId": conversation_id},
            {"_id": 0, "fromFileId": 0, "toFileId": 0},
        ))

        if not rels:
            has_files = db["filePages"].count_documents({"conversationId": conversation_id}, limit=1)
            if has_files:
                build_document_relationships(conversation_id)
                rels = list(db["documentRelationships"].find(
                    {"conversationId": conversation_id},
                    {"_id": 0, "fromFileId": 0, "toFileId": 0},
                ))

        if rels:
            rel_lines = []
            for r in rels:
                rel_lines.append(
                    f"  [{r.get('relationshipType', '').upper()}] "
                    f"{r.get('fromDocumentType', '')} → {r.get('toDocumentType', '')}: "
                    f"{r.get('detail', '')} (resolved: {r.get('resolved', False)})"
                )
            relationship_context = (
                "\n\nDOCUMENT RELATIONSHIP MAP:\n"
                "The following cross-document relationships have been detected in this corpus:\n"
                + "\n".join(rel_lines)
                + "\n\nUse this map to answer questions about which document is current, "
                "which party now holds obligations after a novation, and which agreements "
                "have been terminated.\n"
            )

        entity_summary = get_entity_summary(conversation_id)
        known_persons = {
            k.replace("person:", "").strip()
            for k in entity_summary
            if k.startswith("person:")
        }
        known_orgs = {
            k.replace("organization:", "").strip()
            for k in entity_summary
            if k.startswith("organization:")
        }
        query_lower = query.lower()
        matched_person = next((p for p in known_persons if p and p in query_lower), None)

        # Also check organizations — handles "what is Eames Consulting's role?"
        # where the entity is a company, not an individual.
        matched_org = None
        if not matched_person:
            for org in known_orgs:
                if not org:
                    continue
                # Match on any meaningful token (3+ chars) of the org name
                org_tokens = [t for t in re.split(r'\W+', org.lower()) if len(t) >= 3]
                if org_tokens and any(t in query_lower for t in org_tokens):
                    matched_org = org
                    break

        if matched_person:
            person_docs = get_documents_for_person(conversation_id, matched_person)
            if person_docs:
                thread_lines = []
                for d in person_docs:
                    status = (
                        "[CURRENT]" if d.get("isCurrent")
                        else "[TERMINATED]" if d.get("isTerminated")
                        else f"[{d.get('functionalRole', '').upper()}]"
                    )
                    thread_lines.append(
                        f"  {status} {d.get('fileName', '')} "
                        f"(role: {d.get('functionalRole', '?')}, "
                        f"date: {d.get('effectiveDate') or '?'})"
                    )
                entity_thread_context = (
                    f"\n\nENTITY THREAD — Documents mentioning '{matched_person}':\n"
                    + "\n".join(thread_lines)
                    + "\nThe [CURRENT] document is the most recent non-terminated transaction.\n"
                )
        elif matched_org:
            from src.services.entity_extractor import find_files_by_entity
            org_file_ids = find_files_by_entity(conversation_id, matched_org, entity_type="organization")
            if org_file_ids:
                org_pages = list(db["filePages"].find(
                    {"conversationId": conversation_id, "fileId": {"$in": org_file_ids}},
                    {"fileId": 1, "fileName": 1, "functionalRole": 1, "effectiveDate": 1},
                ))
                _role_order = {"master_agreement": 0, "transaction": 1, "modification": 2, "termination": 3}
                org_pages.sort(key=lambda x: (
                    _role_order.get(x.get("functionalRole", "standalone"), 4),
                    x.get("effectiveDate") or "",
                ))
                thread_lines = []
                for p in org_pages:
                    role = p.get("functionalRole", "standalone")
                    thread_lines.append(
                        f"  [{role.upper()}] {p.get('fileName', '')} "
                        f"(date: {p.get('effectiveDate') or '?'})"
                    )
                entity_thread_context = (
                    f"\n\nENTITY THREAD — Documents mentioning '{matched_org}':\n"
                    + "\n".join(thread_lines)
                    + "\nCross-reference ALL listed documents to determine this organization's role and obligations.\n"
                )

    except Exception as rel_e:
        insert_logs(
            message=f"[CI] Non-critical: could not load relationship map: {rel_e}",
            logType="warning",
            bulkId=conversation_id,
        )

    # --- Build answer prompt (identical to askAttachmentsNode) ---
    from datetime import datetime
    _today = datetime.now().strftime("%B %d, %Y")

    prompt_context = f"""Today's date: {_today}

Here are the contents of the attached files:
{relevant_context}
{relationship_context}{entity_thread_context}
Based on the above, please answer the following question:
{query}

CRITICAL INSTRUCTIONS FOR COMPREHENSIVE, DETAILED RESPONSES:

1. **SEARCH THOROUGHLY** - Examine ALL sections including:
   ✓ Main agreement text
   ✓ Schedules, Exhibits, and Appendices (critical for SLAs, pricing, specifications)
   ✓ Tables and structured data
   ✓ Referenced sections and clause numbers

2. **EXTRACT SPECIFIC DETAILS** - Include precise information:
   • Document reference numbers (e.g., "ASTON-22-7464-02", "Agreement Number: XXX")
   • **PAGE NUMBERS** - Use the [Source X: filename.pdf, Page Y] markers to cite pages
   • Exact metrics and targets (e.g., "99.99% availability", "2-hour response time")
   • Specific timeframes and deadlines
   • Financial terms, penalties, or service credits
   • Specific clause locations (e.g., "Schedule 1, Section 4, Clause 8.2")

3. **FOLLOW THE DOCUMENT RELATIONSHIP MAP** — if a DOCUMENT RELATIONSHIP MAP appears above,
   use it to reason about the current state of the corpus:
   • **NOVATION**: After a novation, the incoming party holds all obligations.
     **CRITICAL — TEMPORAL LIABILITY WINDOW**: If the query mentions a specific date or event,
     perform a timeline check BEFORE assigning liability:
     1. Find the Effective Date of every novation in the evidence.
     2. Determine which party was "The Company" / "the Service Provider" ON the date of the event.
     3. Check the novation for a "Retained Liability" or "acts or omissions" clause.
   • **TERMINATION**: A Work Order marked as terminated is no longer in force.
   • **RENEWAL**: When multiple Work Orders exist, the most recent is current unless terminated.
   • **MASTER_CHILD**: Every Work Order is subordinate to the Master Agreement.

4. **APPLY SPECIFICITY HIERARCHY** — specific provisions govern over general ones:
   • Schedule or Appendix clause **overrides** the main agreement body clause
   • A source marked **[HISTORICAL VERSION]** has been superseded — cite for context only
   • A source marked **[effective YYYY-MM-DD]** — the most recent date wins for the same topic

4a. **RISK FLAGS** — the context may include sections marked [⚠ HIGH RISK] or [⚠ ABOVE MARKET]:
   • **HIGH RISK (level 5)**: Immediately flag these. Unlimited indemnity, one-sided liability, IP traps.
   • **ABOVE MARKET (level 4)**: Note as commercially aggressive and explain why.

5. **IDENTIFY HEDGING AND LIMITATION LANGUAGE**:
   • Warranty disclaimers, endeavours qualifiers, target vs guarantee distinctions
   • Override language: "notwithstanding", "subject to", "except where", "provided that"
   • **CRITICAL — "Target" ≠ "Not Found"**: An "Availability Target" is a contractual commitment.

6. **BE ACCURATE AND SPECIFIC**:
   • Do NOT say "not explicitly listed" if information EXISTS in the documents
   • Reference ACTUAL numbers, percentages, and timeframes
   • **ALWAYS cite page numbers** using the [Source X: filename.pdf, Page Y] markers

7. **TEMPORAL STATUS CHECK** — Today's date is {_today}. For EVERY agreement, work order,
   or engagement period found in the documents:
   • Compare its expiry / end date to today and state: ACTIVE, LAPSED, or RENEWED
   • If lapsed with no documented successor: flag "⚠️ ENGAGEMENT LAPSED — no active instrument found as of {_today}"
   • For tiered schedules tied to tenure (e.g. fee waivers after N months): calculate elapsed time
     from commencement to today and state whether the threshold has been crossed

8. **DRAW ANALYTICAL CONCLUSIONS** — Do not stop at reporting facts. After extraction:
   • Apply fee/waiver thresholds: if extracted tenure ≥ threshold stated in the document,
     explicitly conclude the benefit IS or IS NOT triggered (e.g. "conversion fee waived")
   • Apply novation logic: state who holds obligations TODAY, not just at signing
   • Apply renewal logic: if the latest WO has lapsed and no WO4 exists, conclude the
     engagement has ended and flag the commercial risk

9. **REQUIRED DOCUMENTS CHECK** — If any agreement references a secondary document that
   must be executed (Deed Poll, Data Collection Statement, IP Assignment, Side Letter, etc.):
   • Check whether that document appears among the provided sources by name
   • If present: confirm and cite it
   • If absent: flag "⚠️ [Document name] is required by [clause] but is NOT present in the
     document set — existence cannot be confirmed from available evidence"
   • Do NOT conclude it was never signed — only note it is not in the current set

RETURN FORMAT - Return in JSON:
```json
{{
  "answer": "Structure your answer exactly as a senior partner briefing a client:\\n\\n[OPEN with Risk Flags if ANY ⚠ sources are in context]\\n\\n---\\n\\n## Current Position\\n*The governing rule as it stands today ({_today}). Include ACTIVE / LAPSED / RENEWED status for every instrument.*\\n\\n---\\n\\n## Full Extraction\\n*Every specific number, percentage, timeframe, and qualifier found.*\\n\\n---\\n\\n## Analytical Conclusions\\n*Logical inferences drawn from the extracted facts: thresholds crossed, fees waived, engagements lapsed, obligations transferred.*\\n\\n---\\n\\n## Expert Observations\\n*What a 30-year partner would flag — gaps, risks, market deviations, required documents not in set.*\\n\\n---\\n\\n## Summary Table\\n\\n| Component | Current Rule | Source | Status as of {_today} | Risk |\\n|-----------|-------------|--------|----------------------|------|",
  "citations": [
    {{
      "sourceIndex": 1,
      "TextToFind": "exact text snippet from the document"
    }}
  ],
  "reasoning": "Brief explanation of how you identified the governing provision."
}}
```
"""

    # --- Call answer LLM (identical to askAttachmentsNode) ---
    from src.utils.llm_utils import invoke_answer_with_costing_evaluation
    response = invoke_answer_with_costing_evaluation(prompt=prompt_context)

    if isinstance(response, dict) and "errorMessage" in response:
        raise HTTPException(status_code=500, detail=f"LLM error: {response['errorMessage']}")

    if not hasattr(response, "content"):
        raise HTTPException(status_code=500, detail="LLM response missing content attribute")

    response_content = response.content.strip()

    # --- Parse JSON response + resolve citations (identical to askAttachmentsNode) ---
    try:
        if "```json" in response_content:
            json_start = response_content.find("```json") + 7
            json_end = response_content.find("```", json_start)
            json_str = response_content[json_start:json_end].strip()
        else:
            json_str = response_content

        structured = json.loads(json_str)

        processed_citations = []
        for citation in structured.get("citations", []):
            source_idx = citation.get("sourceIndex")
            if source_idx and source_idx in context_metadata:
                meta = context_metadata[source_idx]
                processed_citations.append({
                    "fileId": meta["fileId"],
                    "fileName": meta["fileName"],
                    "PageNumber": meta["pageNumber"],
                    "TextToFind": citation.get("TextToFind", ""),
                })

        answer = structured.get("answer", response_content)

    except (json.JSONDecodeError, KeyError):
        # Fallback: plain text answer + inline citation extraction
        answer = response_content
        processed_citations = []
        _source_re = re.compile(r'Source:\s*([^,\n]+?),\s*Page\s*(\d+)', re.IGNORECASE)
        seen: set = set()
        for match in _source_re.finditer(response_content):
            raw_name = match.group(1).strip()
            page_num = match.group(2).strip()
            for _, meta in context_metadata.items():
                if raw_name.lower() in meta["fileName"].lower():
                    key = (meta["fileName"], page_num)
                    if key not in seen:
                        seen.add(key)
                        processed_citations.append({
                            "fileId": meta["fileId"],
                            "fileName": meta["fileName"],
                            "PageNumber": page_num,
                            "TextToFind": "",
                        })
                    break

    insert_logs(
        message=f"[CI] Query answered with {len(processed_citations)} citation(s)",
        logType="information",
        bulkId=conversation_id,
    )

    return {
        "answer": answer,
        "citations": processed_citations,
    }


# ---------------------------------------------------------------------------
# GET /ci/file/{fileId}
# ---------------------------------------------------------------------------

@router.get("/file/{file_id}", summary="Serve a contract file for the PDF viewer")
def serve_file(file_id: str):
    """
    Returns the raw file so the frontend PDF viewer can open it inline.
    Resolves the local path from filePages.localPath.
    """
    local_path = _get_local_path(file_id)
    if not local_path or not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=local_path,
        media_type="application/pdf",
        filename=os.path.basename(local_path),
    )


# ---------------------------------------------------------------------------
# GET /ci/conversations/{conversationId}/files
# ---------------------------------------------------------------------------

@router.get(
    "/conversations/{conversation_id}/files",
    summary="List all files in a conversation",
)
def list_conversation_files(conversation_id: str):
    """
    Returns all files uploaded to a conversation with their processing status.
    Used by the frontend to restore state on page refresh.
    """
    records = list(db["filePages"].find(
        {"conversationId": conversation_id},
        {"fileId": 1, "fileName": 1, "processing_status": 1, "sections_embedded": 1},
    ))
    return [
        {
            "fileId": r["fileId"],
            "fileName": r.get("fileName", ""),
            "processing_status": r.get("processing_status", "unknown"),
            "sections_embedded": r.get("sections_embedded", False),
        }
        for r in records
    ]


# ---------------------------------------------------------------------------
# GET /ci/conversations/{conversationId}/graph
# ---------------------------------------------------------------------------

@router.get(
    "/conversations/{conversation_id}/graph",
    summary="Get knowledge graph for a conversation (overview)",
)
def get_conversation_graph(conversation_id: str):
    """
    Returns the high-level knowledge graph for all documents in a conversation.
    Nodes: documents, persons, organisations.
    Edges: MASTER_OF, NOVATES, TERMINATES, RENEWS, PARTY_TO.
    React-Flow compatible format.
    """
    try:
        from src.services.graph_extractor import get_graph_for_conversation
        return get_graph_for_conversation(conversation_id)
    except Exception as e:
        logger.error("[CI] Graph fetch failed for conversation %s: %s", conversation_id, e)
        raise HTTPException(status_code=500, detail=f"Graph unavailable: {e}")


# ---------------------------------------------------------------------------
# GET /ci/files/{fileId}/graph
# ---------------------------------------------------------------------------

@router.get(
    "/files/{file_id}/graph",
    summary="Get clause-level knowledge graph for one document (drill-down)",
)
def get_file_graph(file_id: str):
    """
    Returns the section-level knowledge graph for a single document.
    Nodes: document, sections (colour-coded by clauseType), parties.
    Edges: CONTAINS, CHILD_OF, REFERENCES, CONDITIONS, SUPERSEDES, EXCEPTIONS,
           DEFINES, OBLIGATES, PERMITS, PROHIBITS, PARTY_TO.
    React-Flow compatible format.
    """
    record = db["filePages"].find_one({"fileId": file_id}, {"conversationId": 1})
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    conversation_id = record.get("conversationId")
    if not conversation_id:
        raise HTTPException(status_code=422, detail="File has no conversationId")

    try:
        from src.services.graph_extractor import get_section_graph
        return get_section_graph(file_id, conversation_id)

    except Exception as e:
        logger.error("[CI] Section graph fetch failed for file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"Graph unavailable: {e}")


# ---------------------------------------------------------------------------
# POST /ci/conversations/{conversationId}/reprocess-graph
# ---------------------------------------------------------------------------

@router.post(
    "/conversations/{conversation_id}/reprocess-graph",
    summary="Re-run entity extraction + graph rebuild for all files in a conversation",
)
def reprocess_graph(conversation_id: str):
    """
    Re-runs entity extraction (with corrected role classification) and rebuilds
    the knowledge graph for every file in the conversation.
    Use this after updating the entity extractor prompt/rules without re-uploading.
    Runs synchronously — may take 30–60 seconds for a large conversation.
    """
    records = list(db["filePages"].find(
        {"conversationId": conversation_id},
        {"fileId": 1, "fileName": 1},
    ))
    if not records:
        raise HTTPException(status_code=404, detail="No files found for this conversation")

    from src.services.entity_extractor import extract_entities
    from src.services.graph_extractor import extract_graph
    from src.services.document_relationship_service import build_document_relationships

    # Step 1 — Re-run entity extraction for all files (corrects functionalRole)
    results = []
    for rec in records:
        file_id = rec["fileId"]
        file_name = rec.get("fileName", file_id)
        try:
            extract_entities(file_id, conversation_id)
            results.append({"fileId": file_id, "fileName": file_name, "status": "entities_ok"})
        except Exception as e:
            logger.error("[CI] Reprocess entity extraction failed for %s: %s", file_name, e)
            results.append({"fileId": file_id, "fileName": file_name, "status": "entities_error", "error": str(e)})

    # Step 2 — Rebuild document relationships (must run after entity extraction)
    try:
        build_document_relationships(conversation_id)
    except Exception as e:
        logger.error("[CI] Reprocess build_document_relationships failed: %s", e)

    # Step 3 — Re-run graph extraction for all files
    final_results = []
    for rec in records:
        file_id = rec["fileId"]
        file_name = rec.get("fileName", file_id)
        try:
            summary = extract_graph(file_id, conversation_id)
            final_results.append({"fileId": file_id, "fileName": file_name, "status": "ok", **summary})
        except Exception as e:
            logger.error("[CI] Reprocess graph failed for %s: %s", file_name, e)
            final_results.append({"fileId": file_id, "fileName": file_name, "status": "graph_error", "error": str(e)})

    return {"reprocessed": len(final_results), "results": final_results}


# ---------------------------------------------------------------------------
# POST /ci/conversations/{conversationId}/reembed
# ---------------------------------------------------------------------------

@router.post(
    "/conversations/{conversation_id}/reembed",
    summary="Re-embed all contract sections into Qdrant for a conversation",
)
def reembed_conversation(conversation_id: str):
    """
    Re-runs section embedding into Qdrant for every file in the conversation.
    Use this when Qdrant sections are missing (system fell back to FULL DOCS mode
    on every query). Safe to call multiple times — uses upsert so no duplicates.

    Returns per-file counts so you can confirm success.
    """
    records = list(db["filePages"].find(
        {"conversationId": conversation_id},
        {"fileId": 1, "fileName": 1},
    ))
    if not records:
        raise HTTPException(status_code=404, detail="No files found for this conversation")

    from src.services.section_embedder import embed_sections_to_qdrant

    results = []
    total = 0
    for rec in records:
        file_id = rec["fileId"]
        file_name = rec.get("fileName", file_id)
        try:
            n = embed_sections_to_qdrant(file_id, conversation_id)
            total += n
            results.append({"fileId": file_id, "fileName": file_name, "sections": n, "status": "ok"})
        except Exception as e:
            logger.error("[CI] Reembed failed for %s: %s", file_name, e)
            results.append({"fileId": file_id, "fileName": file_name, "sections": 0, "status": "error", "error": str(e)})

    return {
        "conversationId": conversation_id,
        "totalSectionsEmbedded": total,
        "files": results,
    }
