"""
Desk Conversation — SSE-based conversation endpoints for the Legal Desk frontend.

Mirrors the main-app /conversation + SSE streaming pattern but uses the local-disk
Contract Intelligence RAG pipeline instead of Azure Blob.

Endpoints:
    POST /desk/conversation                          — create/continue thread, kick off RAG
    GET  /desk/messages/stream/{aiMessageId}         — SSE: stream AI response when ready
    GET  /messages/stream/{conversationId}/{msgId}   — SSE: stream agent progress messages
    GET  /messages/thread/{conversationId}           — load historical messages for a thread

Flow:
    1. Frontend POST /desk/conversation → gets { conversationId, messageId, aiMessageId }
    2. Frontend opens EventSource /desk/messages/stream/{aiMessageId}
    3. Frontend opens EventSource /messages/stream/{conversationId}/{messageId}
    4. Backend runs RAG in background, writes result to messages collection
    5. SSE stream detects completion → sends ai-response-update with isComplete=True
"""

import asyncio
import datetime
import json
import logging
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.utils.connection_utils import db
from src.utils.log_utils import insert_logs

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DeskConversationRequest(BaseModel):
    query: str
    conversationId: Optional[str] = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _clean(doc: dict) -> dict:
    """Remove MongoDB _id and convert datetime fields to ISO strings."""
    doc.pop("_id", None)
    for k, v in doc.items():
        if isinstance(v, datetime.datetime):
            doc[k] = v.isoformat()
    return doc


def _ai_message_payload(record: dict, is_complete: bool = False) -> dict:
    """Normalise an AI message document for SSE delivery."""
    doc = _clean(dict(record))
    doc["isComplete"] = is_complete
    return doc


def _agent_msg_payload(msg: dict) -> dict:
    """Normalise a single agent message for SSE delivery."""
    msg = dict(msg)
    msg.pop("latestTimestamp", None)
    return msg


# ---------------------------------------------------------------------------
# Background RAG task
# ---------------------------------------------------------------------------

def _push_agent_message(message_id: str, text: str, agent_name: str = "Contract Intelligence"):
    """Append a progress message to the agentmessages collection."""
    try:
        db["agentmessages"].update_one(
            {"messageId": message_id},
            {
                "$push": {
                    "agentMessage": {
                        "processMessage": text,
                        "agentName": agent_name,
                        "showOnUI": True,
                        "latestTimestamp": datetime.datetime.now(datetime.timezone.utc),
                    }
                },
                "$setOnInsert": {"messageId": message_id, "isCompleted": False},
            },
            upsert=True,
        )
    except Exception as e:
        logger.warning("[Desk] agent message push failed: %s", e)


def _run_rag_background(
    conversation_id: str,
    query: str,
    message_id: str,
    ai_message_id: str,
):
    """
    Full RAG pipeline for a desk conversation message.
    Runs in FastAPI BackgroundTasks thread pool.
    """
    try:
        insert_logs(
            message=f"[Desk] RAG started for conversation {conversation_id}",
            logType="information",
            bulkId=conversation_id,
        )

        # Mark AI message as Processing
        db["messages"].update_one(
            {"messageId": ai_message_id},
            {"$set": {"status": "Processing"}},
        )

        _push_agent_message(message_id, "Analysing your question…")

        # --- Wait for all files in this conversation to finish processing ---
        # Scanned PDFs can take several minutes (OCR page-by-page). We poll
        # filePages until every file is ready or failed before attempting RAG.
        _push_agent_message(message_id, "Waiting for document processing to complete…")
        _wait_deadline = time.monotonic() + 600  # 10-minute max wait
        while time.monotonic() < _wait_deadline:
            pending = db["filePages"].count_documents({
                "conversationId": conversation_id,
                "processing_status": {"$in": ["queued", "pre_processing"]},
            })
            if pending == 0:
                break
            time.sleep(5)

        # --- Retrieve sections ---
        from src.services.section_retriever import retrieve_sections, all_sections_embedded

        _push_agent_message(message_id, "Retrieving relevant contract sections…")

        if all_sections_embedded(conversation_id):
            try:
                relevant_context, context_metadata = retrieve_sections(conversation_id, query)
                if not relevant_context:
                    raise ValueError("Empty context from section retriever")
            except Exception as retr_e:
                insert_logs(
                    message=f"[Desk] Retriever fallback: {retr_e}",
                    logType="warning",
                    bulkId=conversation_id,
                )
                from src.utils.full_document_loader import load_full_documents_with_citations
                relevant_context, context_metadata = load_full_documents_with_citations(conversation_id)
        else:
            from src.utils.full_document_loader import load_full_documents_with_citations
            relevant_context, context_metadata = load_full_documents_with_citations(conversation_id)

        if not relevant_context:
            raise ValueError("Could not retrieve document content — check files are fully processed")

        # --- Relationship / entity context (best-effort) ---
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
                rel_lines = [
                    f"  [{r.get('relationshipType','').upper()}] "
                    f"{r.get('fromDocumentType','')} → {r.get('toDocumentType','')}:"
                    f" {r.get('detail','')} (resolved: {r.get('resolved',False)})"
                    for r in rels
                ]
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
            query_lower = query.lower()
            matched_person = next((p for p in known_persons if p and p in query_lower), None)
            if matched_person:
                person_docs = get_documents_for_person(conversation_id, matched_person)
                if person_docs:
                    thread_lines = [
                        f"  {'[CURRENT]' if d.get('isCurrent') else '[TERMINATED]' if d.get('isTerminated') else '[' + d.get('functionalRole','').upper() + ']'}"
                        f" {d.get('fileName','')} (role: {d.get('functionalRole','?')}, date: {d.get('effectiveDate') or '?'})"
                        for d in person_docs
                    ]
                    entity_thread_context = (
                        f"\n\nENTITY THREAD — Documents mentioning '{matched_person}':\n"
                        + "\n".join(thread_lines)
                        + "\nThe [CURRENT] document is the most recent non-terminated transaction.\n"
                    )
        except Exception as rel_e:
            insert_logs(
                message=f"[Desk] Non-critical: relationship map error: {rel_e}",
                logType="warning",
                bulkId=conversation_id,
            )

        # --- Load conversation history (last 10 Q&A pairs for context) ---
        _push_agent_message(message_id, "Loading conversation context…")
        conversation_history = ""
        try:
            prior_messages = list(
                db["messages"]
                .find(
                    {
                        "conversationId": conversation_id,
                        "status": "Processed",
                        "messageId": {"$ne": ai_message_id},
                    },
                    {"_id": 0, "role": 1, "content": 1, "createdAt": 1},
                )
                .sort("createdAt", 1)
                .limit(20)  # up to 10 Q&A pairs
            )

            history_lines = []
            for m in prior_messages:
                role = m.get("role", "")
                content = m.get("content", "")
                # content can be a string, a list of {type,content} dicts, or empty
                if isinstance(content, list):
                    text = " ".join(
                        c.get("content", "") for c in content
                        if isinstance(c, dict) and c.get("content")
                    )
                else:
                    text = str(content) if content else ""

                if role == "user" and text:
                    history_lines.append(f"human: {text.strip()}")
                elif role == "ai" and text:
                    # Truncate long AI answers to keep prompt size reasonable
                    truncated = text.strip()[:800] + ("…" if len(text) > 800 else "")
                    history_lines.append(f"assistant: {truncated}")

            if history_lines:
                conversation_history = (
                    "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
                    + "\n".join(history_lines)
                    + "\n\nUse the above conversation history to understand follow-up questions "
                    "and resolve pronouns (e.g. 'they', 'it', 'the clause mentioned above').\n"
                )
        except Exception as hist_e:
            insert_logs(
                message=f"[Desk] Non-critical: conversation history error: {hist_e}",
                logType="warning",
                bulkId=conversation_id,
            )

        # --- Build prompt (identical to askAttachmentsNode) ---
        prompt_context = f"""Here are the contents of the attached files:
{relevant_context}
{relationship_context}{entity_thread_context}{conversation_history}
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
   • **PAGE NUMBERS** - Use the [Source X: filename.pdf, Page Y] markers to cite pages (e.g., "SOW Page 3", "SaaS Agreement Page 12")
   • Exact metrics and targets (e.g., "99.99% availability", "2-hour response time", "4-hour resolution")
   • Specific timeframes and deadlines
   • Severity levels (P1/Critical, P2/Major, P3/Minor)
   • Financial terms, penalties, or service credits
   • Specific clause locations (e.g., "Schedule 1, Section 4, Clause 8.2")

3. **FOLLOW THE DOCUMENT RELATIONSHIP MAP** — if a DOCUMENT RELATIONSHIP MAP appears above,
   use it to reason about the current state of the corpus:
   • **NOVATION**: After a novation, the incoming party holds all obligations. Cite the novation
     agreement as the source of this transfer. If the question asks "who is the current service
     provider?", trace the full novation chain to the final incoming party.

     **CRITICAL — TEMPORAL LIABILITY WINDOW**: If the query mentions a specific date or event,
     you MUST perform a timeline check BEFORE assigning liability:
     1. Find the Effective Date of every novation in the evidence.
     2. Determine which party was "The Company" / "the Service Provider" ON the date of the event.
     3. Check the novation for a "Retained Liability" or "acts or omissions" clause (often §2.2(b)):
        this clause states the OUTGOING party remains liable for all acts, defaults, or breaches
        that occurred BEFORE the novation's Effective Date.
     The current counterparty is ONLY liable for events on or after THEIR novation's Effective Date.
     WRONG: "Eames is liable for a breach on 15 Sept 2023" (when Eames became party on 1 Nov 2023).
     RIGHT: "Mainstay SG is liable — the Eames novation took effect 1 Nov 2023; under the
     retained-liability clause, Mainstay SG remains responsible for all acts before that date."
   • **TERMINATION**: A Work Order marked as terminated is no longer in force. Note the
     termination date and do not treat obligations under it as current.
   • **RENEWAL**: When multiple Work Orders exist for the same consultant, the most recent
     (highest commencement date) is current unless terminated. Label earlier ones as
     "[SUPERSEDED]" and the latest as "[CURRENT]".
   • **MASTER_CHILD**: Every Work Order is subordinate to the Master Agreement. If a question
     about a Work Order is silent on a term, the Master Agreement governs.

4. **STRUCTURE YOUR ANSWER** - Organize with clear visual separation:
   ```
   Yes/No, [brief summary statement].

   The contract documents include [description], divided into:

   ---

   ## 1. [Category Name] (Document Name - Page X)

   **[Subcategory]:**
   • **Key Point 1**: Specific detail with exact numbers/dates (Source: SOW Page 3, Section 1.2)
   • **Key Point 2**: Another detail with metrics (Source: SaaS Agreement Page 12, Schedule 4)
   • **Key Point 3**: Reference to specific clauses

   ---

   ## 2. [Next Category] (Document Name - Pages X-Y)

   **[Subcategory]:**
   • Detail with page reference...

   ---

   ## Summary Comparison Table

   | Service Component | Defined? | Location (Document, Page, Section) |
   |-------------------|----------|-----------------------------------|
   | Item 1            | ✓ Yes    | SOW Page 3, Section 1.2          |
   | Item 2            | ✓ Yes    | SaaS Agreement Page 12, Schedule 4|
   ```

4. **APPLY SPECIFICITY HIERARCHY** — specific provisions govern over general ones:
   When multiple sources address the same topic, the most specific source controls:
   • Schedule or Appendix clause **overrides** the main agreement body clause
   • Addendum (SOW / SaaS Agreement) clause **overrides** Framework Agreement clause
   • A clause that says "notwithstanding clause X" **overrides** clause X for that subject
   • A source marked **[HISTORICAL VERSION]** has been superseded — cite it only as
     historical context, never as the current governing rule.
   • A source marked **[effective YYYY-MM-DD]** — the most recent date wins for the same topic.
   Always cite the most specific and most recent source as the governing rule. Mention older
   or general clauses only as context, and explicitly state that the specific provision prevails.
   Example: "Schedule 7 §4.3 sets 2%/month for Hosting Charges [effective 2023-01-31],
   overriding the general Bank of Scotland +3% rate in Framework Agreement clause 8.2
   [HISTORICAL VERSION] for this charge type."

4a. **RISK FLAGS** — the context may include sections marked [⚠ HIGH RISK] or [⚠ ABOVE MARKET]:
   • **HIGH RISK (level 5)**: Immediately flag these to the user. These clauses contain
     unlimited indemnity, one-sided liability, IP traps, or uncapped exposure.
   • **ABOVE MARKET (level 4)**: Note these as commercially aggressive and explain why.
   • Present a **Risk Summary** at the top of your answer if any risk-flagged sections are in context.
   Example Risk Summary:
   ```
   ⚠ RISK FLAGS IDENTIFIED:
   • [HIGH RISK] §12.3 — Unlimited indemnity on client side with no cap. (SaaS Agreement, Page 18)
   • [ABOVE MARKET] §8.1 — 90-day auto-renewal lock-in with no break right. (Framework, Page 9)
   ```

5. **IDENTIFY HEDGING AND LIMITATION LANGUAGE** — this is critical for legal accuracy:
   Every commitment, metric, or right may be qualified. Before stating any finding as fact, scan
   for language that limits, conditions, or carves out that commitment:
   • **Warranty disclaimers**: "makes no warranty", "cannot guarantee", "does not warrant", "no assurance"
   • **Endeavours qualifiers**: "best endeavours", "reasonable endeavours", "reasonable steps", "strive to"
   • **Target vs guarantee**: if the document says "target" or "aim" rather than "guarantee" or "shall", state it as a target
   • **Override language**: "notwithstanding", "subject to", "except where", "provided that", "unless"
   • **Exclusions and carve-outs**: planned maintenance, force majeure, client-caused downtime, third-party failures
   When a metric IS qualified, report it as: "[Metric] is the [target/cap/aim], subject to [qualification]."
   Never present a target as an unconditional guarantee. Never omit a notwithstanding clause.

   **CRITICAL — "Target" ≠ "Not Found"**: An "Availability Target", "Service Level Target", or
   "Performance Target" is a CONTRACTUAL COMMITMENT even if the document also contains a "no warranty"
   clause. The "no warranty" language limits the legal remedy type — it does NOT mean the figure is
   absent or that there is no obligation. You MUST report the figure and then note the qualification.
   WRONG: "The document does not guarantee 99.99% uptime."
   RIGHT: "The Availability Target is 99.99% (SaaS Agreement §11, Page 11). This is stated as a
   target, not an unconditional warranty (§11.7.6), meaning breach remedies are limited to Service
   Credits rather than damages. Service Credits are deducted from the next invoice or refunded at
   agreement end (§ Schedule 2, Page 16)."

6. **BE ACCURATE AND SPECIFIC**:
   • Do NOT say "not explicitly listed" if information EXISTS in the documents
   • Reference ACTUAL numbers, percentages, and timeframes found
   • **ALWAYS cite page numbers** using the [Source X: filename.pdf, Page Y] markers in the context
   • Example: "The SLA defines 99.99% availability (Source: SaaS Agreement Page 12, Section 11.3)"
   • If comparing multiple documents, create a table showing differences WITH page references

6. **CREATE COMPARISON TABLES** when appropriate to show:
   • What different documents cover
   • Service levels vs. actual metrics
   • Framework Agreement vs. SOW vs. SaaS Agreement

RETURN FORMAT - Return in JSON using the EXPERT AUDIT TRAIL structure:
```json
{{
  "answer": "Structure your answer exactly as a senior partner briefing a client:\\n\\n[OPEN with Risk Flags if ANY ⚠ sources are in context:]\\n⚠ RISK FLAGS:\\n• [HIGH RISK] §X.X — one-line description (Document, Page N)\\n• [ABOVE MARKET] §Y.Y — one-line description (Document, Page N)\\n\\n---\\n\\n## Current Position\\n*The governing rule as it stands today — most recent, most specific provision wins.*\\n\\n**[Topic]:** [Exact obligation / figure] per [Most Specific Source §X.X, Page N, effective YYYY-MM-DD].\\n[If a HISTORICAL source exists for the same topic:] Previously governed by [Older Source §Y.Y, HISTORICAL]; that version no longer applies.\\n\\n---\\n\\n## Version History\\n*How this obligation evolved across the document suite.*\\n\\n| Version | Document | Date | Rule |\\n|---------|----------|------|------|\\n| Current | [Doc] | [Date] | [Rule] |\\n| Prior | [Doc] [HISTORICAL] | [Date] | [Old Rule] |\\n\\n---\\n\\n## Full Extraction\\n*Every specific number, percentage, timeframe, and qualifier found.*\\n\\n**[Sub-topic 1]:**\\n• **[Metric]**: Exact figure (Source: Doc Page N, §X.X)\\n• **[Qualifier]**: Any hedging language — 'target not guarantee', 'subject to clause Y', 'reasonable endeavours'\\n\\n**[Sub-topic 2]:** [Continue for all relevant specifics...]\\n\\n---\\n\\n## Expert Observations\\n*What a 30-year partner would flag — gaps, risks, market deviations.*\\n\\n• **[Gap]**: [e.g., 'No data-breach carve-out from the liability cap — unlimited exposure on a breach.']\\n• **[Risk]**: [e.g., '90-day auto-renewal notice window is above market; standard is 30 days.']\\n• **[Missing Clause]**: [e.g., 'SOW contains no governing law clause — Framework's Singapore jurisdiction applies by default.']\\n\\n---\\n\\n## Summary Table\\n\\n| Component | Current Rule | Source | Effective Date | Risk |\\n|-----------|-------------|--------|----------------|------|\\n| [Item] | [Exact value] | [Doc §X.X, Page N] | [YYYY-MM-DD] | [1-5] |",
  "citations": [
    {{
      "sourceIndex": 1,
      "TextToFind": "exact text snippet from the document"
    }}
  ],
  "reasoning": "I identified the most recent and most specific governing provision first. I annotated HISTORICAL versions. I flagged HIGH RISK and ABOVE MARKET clauses with specific risk notes. I extracted every exact number, date, and qualification. I noted gaps versus market standard practice."
}}
```

QUALITY CHECKLIST - Your answer will be evaluated on specificity and expert depth:
✓ Did I open with Risk Flags if any ⚠ HIGH RISK or ABOVE MARKET sources appear?
✓ Did I identify the CURRENT governing provision (most specific + most recent)?
✓ Did I label HISTORICAL versions and explain they are superseded?
✓ Did I check schedules/appendices first (where SLAs, pricing, and specific terms live)?
✓ **Did I check for hedging language?** ('target' not 'guarantee'? 'notwithstanding'? warranty disclaimers?)
✓ **Did I include EVERY SPECIFIC NUMBER?** (99.99%, 2 hours, £90,000, 1.25%, etc.)
✓ Did I create a Version History table when multiple documents address the same topic?
✓ Did I include Expert Observations — what is MISSING or commercially aggressive?
✓ **Did I include PAGE NUMBERS and EFFECTIVE DATES for all key claims?**
✓ Did I avoid saying "not found" when information exists in the documents?
✓ **Did I use the DOCUMENT RELATIONSHIP MAP?** Traced novation chains, flagged terminated WOs, identified the current Work Order for any named consultant.
✓ **TEMPORAL LIABILITY CHECK**: If the query names a specific date, did I verify which party held obligations ON that date — not just who is current today? Did I check the novation's retained-liability clause for pre-Effective-Date events?
✓ **When something is genuinely absent**, did I distinguish between: (a) "not stated anywhere in the uploaded documents" vs (b) "defined by reference to a Schedule or Order Form that may contain it — check pages X–Y"? Never say "not specified" without indicating WHERE it would typically be found in a contract of this type.
"""

        _push_agent_message(message_id, "Generating answer…")

        from src.utils.llm_utils import invoke_answer_with_costing_evaluation, invoke_with_costing_evalution
        response = invoke_answer_with_costing_evaluation(prompt=prompt_context)

        if isinstance(response, dict) and "errorMessage" in response:
            raise ValueError(f"LLM error: {response['errorMessage']}")
        if not hasattr(response, "content"):
            raise ValueError("LLM response missing content attribute")

        response_content = response.content.strip()

        # --- Parse RAG answer (identical to askAttachmentsNode) ---
        answer = response_content
        processed_citations = []
        reasoning = ""

        try:
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_str = response_content[json_start:json_end].strip()
            else:
                json_str = response_content

            structured = json.loads(json_str)
            answer = structured.get("answer", response_content)
            reasoning = structured.get("reasoning", "")

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

        except (json.JSONDecodeError, KeyError):
            # Fallback: inline citation extraction
            _source_re = re.compile(r'Source:\s*([^,\n]+?),\s*Page\s*(\d+)', re.IGNORECASE)
            _seen_cit: set = set()
            for match in _source_re.finditer(response_content):
                raw_name = match.group(1).strip()
                page_num = match.group(2).strip()
                for _, meta in context_metadata.items():
                    if raw_name.lower() in meta["fileName"].lower() or meta["fileName"].lower().startswith(raw_name.lower()[:15]):
                        key = (meta["fileName"], page_num)
                        if key not in _seen_cit:
                            _seen_cit.add(key)
                            processed_citations.append({
                                "fileId": meta["fileId"],
                                "fileName": meta["fileName"],
                                "PageNumber": page_num,
                                "TextToFind": "",
                            })
                        break

        # --- Format output via DashboardModel (identical to final_output_formatter) ---
        _push_agent_message(message_id, "Formatting response…")
        final_content = answer  # fallback: raw string

        try:
            from src.graph.models.askResponseModel import DashboardModel
            json_schema = DashboardModel.model_json_schema()
            format_prompt = f"""
You are a specialized JSON data transformation AI. Your only function is to convert the provided 'RAW_DATA' into a new JSON structure that perfectly matches the 'TARGET_SCHEMA'.

Your entire response must be a single, valid JSON object or array. Do not output anything else—no comments, no explanations, and no markdown formatting.

-----

### 1. Target Schema Definition

Your output MUST strictly conform to the following Pydantic-generated JSON schema...

{json_schema}

-----

### 2. Input Data and Context
...
* **Raw Data to Format**:

    '''json
    {answer}
    '''
...
"""
            fmt_response = invoke_with_costing_evalution(prompt=format_prompt)
            if hasattr(fmt_response, "content"):
                generation = fmt_response.content.strip().replace("```json", "").replace("```", "").replace("\n", "")
                try:
                    final_content = json.loads(generation)
                except json.JSONDecodeError as _jde:
                    if 'Extra data' in str(_jde) and _jde.pos and _jde.pos > 0:
                        try:
                            final_content = json.loads(generation[:_jde.pos])
                        except json.JSONDecodeError:
                            final_content = answer
                    else:
                        final_content = answer
        except Exception as fmt_e:
            insert_logs(
                message=f"[Desk] Non-critical: output formatter failed: {fmt_e}",
                logType="warning",
                bulkId=conversation_id,
            )

        # --- Generate follow-up questions (identical to follow_up_question_generator) ---
        follow_up_questions = []
        try:
            fuq_history = f"human: {query}\nassistant: {answer[:600]}"
            fuq_prompt = f"""
Instructions:

Analyze the provided "Recent Conversation" below. Generate two (2) distinct, highly concise follow-up questions.

Strict Constraint: Each generated question must be 8 words or less.

Question 1 (Core Detail): Focus on clarifying a key fact or a specific action mentioned.

Question 2 (Next Step/Implication): Focus on the immediate next steps or the primary consequence.

Format your output clearly, labeling each question. Generate the output as a JSON array of question strings.

Recent Conversation: [INSERT CONVERSATION TEXT HERE]

{fuq_history}
---
"""
            fuq_response = invoke_with_costing_evalution(prompt=fuq_prompt)
            if hasattr(fuq_response, "content"):
                fuq_raw = fuq_response.content.strip().replace("```json", "").replace("```", "").replace("\n", "")
                follow_up_questions = json.loads(fuq_raw)
        except Exception as fuq_e:
            insert_logs(
                message=f"[Desk] Non-critical: follow-up generator failed: {fuq_e}",
                logType="warning",
                bulkId=conversation_id,
            )

        # --- Persist result to AI message ---
        db["messages"].update_one(
            {"messageId": ai_message_id},
            {
                "$set": {
                    "content": final_content,
                    "citations": processed_citations,
                    "followUpQuestions": follow_up_questions,
                    "reasoning": reasoning,
                    "status": "Processed",
                    "updatedAt": datetime.datetime.now(datetime.timezone.utc),
                }
            },
        )

        # Mark agent messages as completed
        db["agentmessages"].update_one(
            {"messageId": message_id},
            {"$set": {"isCompleted": True}},
        )

        insert_logs(
            message=f"[Desk] RAG complete for conversation {conversation_id}: {len(processed_citations)} citation(s)",
            logType="information",
            bulkId=conversation_id,
        )

    except Exception as e:
        logger.error("[Desk] RAG background task failed: %s", e)
        insert_logs(
            message=f"[Desk] RAG background task failed for {conversation_id}: {e}",
            logType="error",
            bulkId=conversation_id,
        )
        db["messages"].update_one(
            {"messageId": ai_message_id},
            {
                "$set": {
                    "content": f"An error occurred while processing your query: {str(e)}",
                    "citations": [],
                    "followUpQuestions": [],
                    "status": "Failed",
                    "updatedAt": datetime.datetime.now(datetime.timezone.utc),
                }
            },
        )
        db["agentmessages"].update_one(
            {"messageId": message_id},
            {"$set": {"isCompleted": True}},
            upsert=True,
        )


# ---------------------------------------------------------------------------
# POST /desk/conversation
# ---------------------------------------------------------------------------

@router.post("/desk/conversation", summary="Start or continue a desk conversation")
async def desk_conversation(body: DeskConversationRequest, background_tasks: BackgroundTasks):
    """
    Creates user + AI placeholder messages in MongoDB, then fires the RAG pipeline
    in the background. Returns IDs immediately so the frontend can open SSE streams.

    Body:
        query          — user's question
        conversationId — optional; omit to start a new conversation

    Returns:
        { conversationId, messageId, aiMessageId }
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    conversation_id = body.conversationId or str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    ai_message_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc)

    # Ensure conversation record exists
    db["conversations"].update_one(
        {"conversationId": conversation_id},
        {"$setOnInsert": {"conversationId": conversation_id, "createdAt": now}},
        upsert=True,
    )

    # User message
    db["messages"].insert_one({
        "messageId": message_id,
        "aiMessageId": ai_message_id,
        "conversationId": conversation_id,
        "role": "user",
        "content": [{"type": "text", "content": body.query}],
        "status": "Pending",
        "createdAt": now,
    })

    # AI placeholder (will be updated by background task)
    db["messages"].insert_one({
        "messageId": ai_message_id,
        "conversationId": conversation_id,
        "role": "ai",
        "content": [],
        "citations": [],
        "followUpQuestions": [],
        "status": "Pending",
        "createdAt": now,
    })

    background_tasks.add_task(
        _run_rag_background,
        conversation_id,
        body.query,
        message_id,
        ai_message_id,
    )

    return {
        "conversationId": conversation_id,
        "messageId": message_id,
        "aiMessageId": ai_message_id,
    }


# ---------------------------------------------------------------------------
# GET /desk/messages/stream/{ai_message_id}  — AI response SSE
# ---------------------------------------------------------------------------

@router.get("/desk/messages/stream/{ai_message_id}", summary="SSE stream for AI response")
async def desk_message_stream(ai_message_id: str):
    """
    Server-Sent Events stream for a single AI message.

    Events:
        initial-state      — current document state (may be empty/pending)
        ai-response-update — when status changes; isComplete=True signals end

    The client (EventSource) should close after receiving isComplete=True.
    """

    async def event_generator():
        # Yield initial state immediately
        record = db["messages"].find_one({"messageId": ai_message_id}, {"_id": 0})
        if record:
            payload = _ai_message_payload(record, is_complete=False)
            yield f"event: initial-state\ndata: {json.dumps(payload)}\n\n"
        else:
            yield f"event: initial-state\ndata: {json.dumps({'messageId': ai_message_id, 'status': 'Pending'})}\n\n"

        # Poll until the background task writes a final status
        deadline = time.monotonic() + 780  # 13-minute timeout (covers OCR + RAG + LLM)
        last_status = "Pending"

        while time.monotonic() < deadline:
            await asyncio.sleep(0.75)

            record = db["messages"].find_one({"messageId": ai_message_id}, {"_id": 0})
            if not record:
                continue

            status = record.get("status", "Pending")
            if status in ("Processed", "Failed") and status != last_status:
                payload = _ai_message_payload(record, is_complete=True)
                yield f"event: ai-response-update\ndata: {json.dumps(payload)}\n\n"
                return

            last_status = status

        # Timeout — tell the client to stop waiting
        yield f"event: ai-response-update\ndata: {json.dumps({'messageId': ai_message_id, 'status': 'Failed', 'content': 'Request timed out.', 'isComplete': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /messages/stream/{conversation_id}/{message_id}  — Agent progress SSE
# ---------------------------------------------------------------------------

@router.get(
    "/messages/stream/{conversation_id}/{message_id}",
    summary="SSE stream for agent progress messages",
)
async def agent_message_stream(conversation_id: str, message_id: str):  # noqa: ARG001
    """
    Server-Sent Events stream for agent progress messages tied to a user message.

    Events:
        initial-state  — list of agent messages already recorded
        agent-update   — single new agent message as processing progresses

    Stream closes automatically once the agentmessages record has isCompleted=True.
    """

    async def event_generator():
        # Initial snapshot
        record = db["agentmessages"].find_one({"messageId": message_id})
        existing = record.get("agentMessage", []) if record else []
        initial_data = [_agent_msg_payload(m) for m in existing]
        yield f"event: initial-state\ndata: {json.dumps(initial_data)}\n\n"

        seen_count = len(existing)
        deadline = time.monotonic() + 120

        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)

            record = db["agentmessages"].find_one({"messageId": message_id})
            if not record:
                # Task hasn't written anything yet — keep waiting
                continue

            all_msgs = record.get("agentMessage", [])
            if len(all_msgs) > seen_count:
                for msg in all_msgs[seen_count:]:
                    yield f"event: agent-update\ndata: {json.dumps(_agent_msg_payload(msg))}\n\n"
                seen_count = len(all_msgs)

            if record.get("isCompleted"):
                return

        # Timeout — silently close
        return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /messages/thread/{conversation_id}  — Historical messages
# ---------------------------------------------------------------------------

@router.get(
    "/messages/thread/{conversation_id}",
    summary="Load all messages for a conversation thread",
)
def get_thread_messages(conversation_id: str):
    """
    Returns all messages for a conversation, ordered chronologically.
    Used by the frontend to restore state after a page refresh.

    Returns a list of message objects:
        { messageId, aiMessageId?, role, content, citations, followUpQuestions, status, createdAt }
    """
    messages = list(
        db["messages"]
        .find({"conversationId": conversation_id}, {"_id": 0})
        .sort("createdAt", 1)
    )

    result = []
    for m in messages:
        # Convert datetime → ISO string
        for field in ("createdAt", "updatedAt"):
            if isinstance(m.get(field), datetime.datetime):
                m[field] = m[field].isoformat()
        result.append(m)

    return result
