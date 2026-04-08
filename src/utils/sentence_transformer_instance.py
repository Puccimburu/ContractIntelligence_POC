"""
Singleton sentence transformer for section embedding (Phase 2 RAG).

Loads from models/section_embedder/ (local, offline) if present,
otherwise falls back to HuggingFace Hub (requires internet).

Run  python -m scripts.download_sentence_transformer  once to cache locally.
"""

import logging
import pathlib
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_LOCAL_PATH = pathlib.Path("models/section_embedder")
_HF_MODEL = "all-MiniLM-L6-v2"

_model = None


def get_sentence_transformer() -> SentenceTransformer:
    """Lazy-load the sentence transformer singleton."""
    global _model
    if _model is None:
        path = str(_LOCAL_PATH) if _LOCAL_PATH.exists() else _HF_MODEL
        logger.info(f"[SentenceTransformer] Loading from {path} ...")
        _model = SentenceTransformer(path)
        logger.info("[SentenceTransformer] Ready.")
    return _model
