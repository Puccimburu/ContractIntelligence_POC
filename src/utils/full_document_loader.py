"""
Full document loader for comprehensive context retrieval.

Loads all pages from all documents in a conversation, with proper citation metadata.
Uses caching to minimize cost on repeated queries.
"""


import json
from typing import Dict, List, Tuple
from src.utils.db_utils import get_fileInfo_from_conversation, get_collection_details_by_key
from src.utils.log_utils import insert_logs

def load_full_documents_with_citations(conversation_id: str) -> Tuple[str, Dict]:
    """
    Load all pages from all documents in a conversation with citation metadata.

    Returns:
        Tuple of (full_context_text, citation_metadata)

    """

    # CACHING DISABLED - Uncomment below to enable caching
    # cached_result = get_cached_documents(conversation_id)
    # if cached_result:
    #     insert_logs(
    #         message=f"[FULL DOCS] Using cached documents for conversation {conversation_id}",
    #         logType='information',
    #         bulkId=conversation_id
    #     )
    #     return cached_result

    insert_logs(
        message=f"[FULL DOCS] Loading all documents from DB for conversation {conversation_id}",
        logType='information',
        bulkId=conversation_id
    )

    # Get all files in conversation
    files = get_fileInfo_from_conversation(conversation_id)

    if not files:
        insert_logs(
            message=f"[FULL DOCS] No files found for conversation {conversation_id}",
            logType='warning',
            bulkId=conversation_id
        )
        return "", {}

    insert_logs(
        message=f"[FULL DOCS] Found {len(files)} files to load",
        logType='information',
        bulkId=conversation_id
    )

    context_parts = []
    citation_metadata = {}
    source_index = 1

    total_pages = 0
    total_chars = 0

    for file in files:
        file_id = file.get("fileId")
        file_name = file.get("fileName", "Unknown")

        insert_logs(
            message=f"[FULL DOCS] Loading file: {file_name} (fileId: {file_id})",
            logType='information',
            bulkId=conversation_id
        )

        # Get page-wise text from filePages collection
        file_page_doc = get_collection_details_by_key("filePages", "fileId", file_id)

        if not file_page_doc:
            insert_logs(
                message=f"[FULL DOCS] No pages found for fileId: {file_id}",
                logType='warning',
                bulkId=conversation_id
            )
            continue

        page_wise_text_raw = file_page_doc.get("pageWiseText", {})

        # Parse if it's a JSON string (MongoDB sometimes returns as string)
        if isinstance(page_wise_text_raw, str):
            try:
                page_wise_text = json.loads(page_wise_text_raw)
            except json.JSONDecodeError:
                insert_logs(
                    message=f"[FULL DOCS] Failed to parse pageWiseText JSON for fileId: {file_id}",
                    logType='error',
                    bulkId=conversation_id
                )
                page_wise_text = {}
        else:
            page_wise_text = page_wise_text_raw

        if not page_wise_text:
            insert_logs(
                message=f"[FULL DOCS] Empty pageWiseText for fileId: {file_id}",
                logType='warning',
                bulkId=conversation_id
            )
            continue

        # Sort pages by page number
        sorted_pages = sorted(page_wise_text.items(), key=lambda x: int(x[0]))

        file_page_count = 0

        for page_num, page_text in sorted_pages:
            if not page_text or not page_text.strip():
                continue

            # Add to context with citation marker
            context_parts.append(
                f"[Source {source_index}: {file_name}, Page {page_num}]\n{page_text}\n"
            )

            # Store metadata for citation mapping
            citation_metadata[source_index] = {
                "fileId": file_id,
                "fileName": file_name,
                "pageNumber": str(page_num)
            }

            source_index += 1
            file_page_count += 1
            total_chars += len(page_text)

        total_pages += file_page_count

        insert_logs(
            message=f"[FULL DOCS] Loaded {file_page_count} pages from {file_name}",
            logType='information',
            bulkId=conversation_id
        )

    full_context = "\n".join(context_parts)

    insert_logs(
        message=f"[FULL DOCS] Total loaded: {len(files)} files, {total_pages} pages, {total_chars:,} characters, {source_index-1} sources",
        logType='information',
        bulkId=conversation_id
    )

    # Estimate token count (rough: ~4 chars per token)
    estimated_tokens = total_chars // 4
    insert_logs(
        message=f"[FULL DOCS] Estimated tokens: ~{estimated_tokens:,} tokens",
        logType='information',
        bulkId=conversation_id
    )

    # CACHING DISABLED  - Uncomment below to enable caching
    # cache_documents(conversation_id, full_context, citation_metadata)

    return full_context, citation_metadata


def get_document_summary(conversation_id: str) -> Dict:
    """
    Get a summary of documents in a conversation without loading full content.
    Useful for logging and diagnostics.
    """

    files = get_fileInfo_from_conversation(conversation_id)

    summary = {
        "conversation_id": conversation_id,
        "file_count": len(files),
        "files": []
    }

    for file in files:
        file_id = file.get("fileId")
        file_name = file.get("fileName", "Unknown")

        file_page_doc = get_collection_details_by_key("filePages", "fileId", file_id)
        page_wise_text_raw = file_page_doc.get("pageWiseText", {}) if file_page_doc else {}

        # Parse if it's a JSON string
        if isinstance(page_wise_text_raw, str):
            try:
                page_wise_text = json.loads(page_wise_text_raw)
            except json.JSONDecodeError:
                page_wise_text = {}
        else:
            page_wise_text = page_wise_text_raw

        page_count = len(page_wise_text)
        char_count = sum(len(text) for text in page_wise_text.values() if text)

        summary["files"].append({
            "file_id": file_id,
            "file_name": file_name,
            "page_count": page_count,
            "character_count": char_count,
            "estimated_tokens": char_count // 4
        })

    summary["total_pages"] = sum(f["page_count"] for f in summary["files"])
    summary["total_characters"] = sum(f["character_count"] for f in summary["files"])
    summary["estimated_tokens"] = summary["total_characters"] // 4

    return summary
