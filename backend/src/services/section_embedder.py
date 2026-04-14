"""
Section embedder — reads fileSections from MongoDB and upserts vector
embeddings into the 'contract_sections' Qdrant collection.

embed_sections_to_qdrant(file_id, conversation_id)
    Public entry point called from _presubmit_attachment (worker) and
    processAttachmentNode after run_section_parser completes.
    Returns the number of sections embedded. Non-critical on failure.

ensure_section_collection()
    Creates the Qdrant collection once at startup. Called from main.py lifespan.
"""
import logging
import uuid
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

SECTION_COLLECTION = "contract_sections"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


def ensure_section_collection():
    """
    Create the contract_sections Qdrant collection if it does not exist.
    Safe to call multiple times.
    """
    from src.utils.qdrant.qdrant_utils import get_qdrant_client
    from qdrant_client.http.models import VectorParams, Distance

    try:
        client = get_qdrant_client()
        existing = {c.name for c in client.get_collections().collections}
        if SECTION_COLLECTION not in existing:
            client.create_collection(
                collection_name=SECTION_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            # Keyword indexes for fast conversation/file filtering
            client.create_payload_index(
                collection_name=SECTION_COLLECTION,
                field_name="conversationId",
                field_type="keyword",
            )
            client.create_payload_index(
                collection_name=SECTION_COLLECTION,
                field_name="fileId",
                field_type="keyword",
            )
            logger.info(f"[SectionEmbedder] Created Qdrant collection '{SECTION_COLLECTION}'")
        else:
            logger.debug(f"[SectionEmbedder] Collection '{SECTION_COLLECTION}' already exists")
    except Exception as e:
        logger.error(f"[SectionEmbedder] ensure_section_collection failed: {e}")


def _section_point_id(file_id: str, section_id: str) -> str:
    """Stable UUID for a (fileId, sectionId) pair — allows safe re-upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_id}:{section_id}"))


def embed_sections_to_qdrant(file_id: str, conversation_id: str) -> int:
    """
    Read fileSections for file_id from MongoDB, embed with all-MiniLM-L6-v2,
    and upsert into the contract_sections Qdrant collection.

    Returns the number of sections embedded (0 on failure).
    """
    from src.utils.connection_utils import db
    from src.utils.sentence_transformer_instance import get_sentence_transformer
    from src.utils.qdrant.qdrant_utils import get_qdrant_client
    from src.utils.db_utils import upsertCollection
    from src.utils.log_utils import insert_logs
    from qdrant_client.http.models import PointStruct

    _ROLE_RANK = {
        "master_agreement": 0,
        "standalone": 0,
        "modification": 1,
        "transaction": 2,
        "termination": 3,
    }

    try:
        sections = list(db["fileSections"].find({"fileId": file_id}))
        if not sections:
            logger.warning(f"[SectionEmbedder] No sections to embed for fileId={file_id}")
            # Mark as done so the backfill does not pick this file up again
            upsertCollection("filePages", "fileId", file_id, {"sections_embedded": True})
            return 0

        # Read file-level metadata once so every section carries functionalRoleRank,
        # documentRank, and effectiveDate in its Qdrant payload. Phase 1 retrieval
        # depends on these for correct specific-over-general ordering even on the
        # very first query (before _build_file_meta_map has a chance to run).
        file_page = db["filePages"].find_one({"fileId": file_id}) or {}
        functional_role = file_page.get("functionalRole", "standalone")
        if functional_role not in _ROLE_RANK:
            functional_role = "standalone"
        functional_role_rank = _ROLE_RANK[functional_role]
        document_rank = file_page.get("documentRank", 1)
        effective_date = file_page.get("effectiveDate", None)

        model = get_sentence_transformer()
        client = get_qdrant_client()

        texts = [s.get("content", "") for s in sections]
        vectors = model.encode(texts, batch_size=32, show_progress_bar=False)

        points: List[PointStruct] = []
        for sec, vec in zip(sections, vectors):
            points.append(PointStruct(
                id=_section_point_id(file_id, sec["sectionId"]),
                vector=vec.tolist(),
                payload={
                    "fileId": sec["fileId"],
                    "conversationId": sec["conversationId"],
                    "sectionId": sec["sectionId"],
                    "sectionTitle": sec.get("sectionTitle", ""),
                    "pageNumber": sec.get("pageNumber", 0),
                    "fileName": sec.get("fileName", ""),
                    "parentSectionId": sec.get("parentSectionId"),
                    "clauseType": sec.get("clauseType", "other"),
                    "functionalRole": functional_role,
                    "functionalRoleRank": functional_role_rank,
                    "documentRank": document_rank,
                    "effectiveDate": effective_date,
                },
            ))

        # Upsert in batches of 25 to avoid write timeouts on remote Qdrant.
        # Retry each batch on transient SSL/connection drops (same issue seen
        # in section_retriever.py — SSLV3_ALERT_BAD_RECORD_MAC, server disconnect).
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _upsert_batch(batch):
            client.upsert(
                collection_name=SECTION_COLLECTION,
                points=batch,
                wait=True,
            )

        for i in range(0, len(points), 25):
            _upsert_batch(points[i:i + 25])

        # Verify Qdrant actually persisted the points before marking as embedded.
        # Transient connectivity issues can cause the HTTP request to appear to
        # succeed (200 OK) while data is never written. Catching it here prevents
        # sections_embedded=True being set on a conversation with no queryable data.
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        verify = client.count(
            collection_name=SECTION_COLLECTION,
            count_filter=Filter(must=[
                FieldCondition(key="fileId", match=MatchValue(value=file_id))
            ]),
            exact=True,
        )
        if verify.count == 0:
            logger.error(
                f"[SectionEmbedder] Upsert returned OK but Qdrant count=0 for "
                f"fileId={file_id}. sections_embedded NOT set — will retry on next request."
            )
            return 0

        upsertCollection("filePages", "fileId", file_id, {"sections_embedded": True})

        insert_logs(
            message=f"[SectionEmbedder] Embedded {len(points)} section(s) for fileId={file_id}",
            logType='information',
            bulkId=conversation_id,
        )
        return len(points)

    except Exception as e:
        logger.error(f"[SectionEmbedder] embed_sections_to_qdrant failed for fileId={file_id}: {e}")
        return 0
