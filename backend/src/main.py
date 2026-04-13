import time
import os
import uvicorn
import asyncio
import threading
import logging

from dotenv import load_dotenv
load_dotenv()
ENV = os.getenv("APP_ENV", "development")
print(f"🔧 Application starting in '{ENV}' environment")
load_dotenv(f".env.{ENV}", override=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

from src.utils.generic_Utils import CONFIG
from src.utils.log_utils import insert_logs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Suppress noisy third-party debug output
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

TOTAL_CONCURRENT_LIMIT = CONFIG["TOTAL_CONCURRENT_LIMIT"]
SCANNED_PDF_CONCURRENT_LIMIT = CONFIG["SCANNED_PDF_CONCURRENT_LIMIT"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.total_semaphore = asyncio.Semaphore(TOTAL_CONCURRENT_LIMIT)
    app.state.scanned_pdf_semaphore = asyncio.Semaphore(SCANNED_PDF_CONCURRENT_LIMIT)
    insert_logs(
        message=f"Startup: total_semaphore={TOTAL_CONCURRENT_LIMIT}, scanned_pdf_semaphore={SCANNED_PDF_CONCURRENT_LIMIT}",
        logType='information',
    )

    # Ensure contract_sections Qdrant collection exists (Legal Desk RAG)
    try:
        from src.services.section_embedder import ensure_section_collection
        ensure_section_collection()
        insert_logs(message="[Startup] contract_sections collection ensured", logType='information')
    except Exception as e:
        insert_logs(message=f"[Startup] contract_sections warning (non-critical): {e}", logType='warning')

    # MongoDB indexes for section retrieval and cross-references
    try:
        from src.utils.connection_utils import db as _db
        _db['fileSections'].create_index([('fileId', 1), ('sectionId', 1)], background=True)
        _db['fileSections'].create_index([('conversationId', 1), ('sectionId', 1)], background=True)
        _db['fileCrossRefs'].create_index([('conversationId', 1), ('resolvedFileId', 1)], background=True)
        _db['fileCrossRefs'].create_index([('sourceFileId', 1), ('sourceSectionId', 1)], background=True)
        insert_logs(message="[Startup] fileSections/fileCrossRefs indexes ensured", logType='information')
    except Exception as e:
        insert_logs(message=f"[Startup] Index creation warning (non-critical): {e}", logType='warning')

    # Warm up BERT clause classifier (RAG Phase 1)
    try:
        from src.utils.clause_classifier_instance import warmup as _classifier_warmup
        threading.Thread(target=_classifier_warmup, daemon=True).start()
    except Exception as e:
        insert_logs(message=f"[Startup] Clause classifier warmup warning (non-critical): {e}", logType='warning')

    # Warm up cross-encoder re-ranker (RAG Phase 3)
    try:
        from src.utils.cross_encoder_instance import warmup as _reranker_warmup
        threading.Thread(target=_reranker_warmup, daemon=True).start()
    except Exception as e:
        insert_logs(message=f"[Startup] Cross-encoder warmup warning (non-critical): {e}", logType='warning')

    yield

    insert_logs(message="Shutdown complete.", logType="information")


app = FastAPI(
    title="Contract Intelligence — Legal Desk API",
    description="RAG pipeline for legal contract Q&A with section retrieval and citation.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Security middleware ──────────────────────────────────────────────────────

trusted_hosts = CONFIG.get("TRUSTED_HOSTS", "localhost").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

https_redirect = CONFIG.get("ENABLE_HTTPS_REDIRECT", False)
if isinstance(https_redirect, str):
    https_redirect = https_redirect.lower() == "true"
if https_redirect:
    app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["CORSMIDDLEWARE_ORIGINS_ADDRESSES"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response

# ── Local storage directory ──────────────────────────────────────────────────

os.makedirs(os.path.join(os.getcwd(), CONFIG["DOCUMENTS_CONTAINER_DIR"]), exist_ok=True)

# ── Routers ──────────────────────────────────────────────────────────────────

_t = time.perf_counter()
from src.apis import default, contractIntelligence, deskConversation
app.include_router(default.router)
app.include_router(contractIntelligence.router)
app.include_router(deskConversation.router)
print(f"Router import time: {time.perf_counter() - _t:.4f}s")

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("APP_ENV") == "development",
        reload_dirs=["src"],
    )
