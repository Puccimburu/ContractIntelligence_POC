"""
Singleton cross-encoder for Phase 3 section re-ranking.

Loads from models/section_reranker/ (local, offline) if present,
otherwise falls back to HuggingFace Hub (requires internet).
"""

import logging
import pathlib

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_LOCAL_PATH = pathlib.Path("models/section_reranker")
_HF_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


class _CrossEncoder:
    """Minimal CrossEncoder wrapper — exposes .predict(pairs) matching sentence_transformers API."""

    def __init__(self, path: str, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def predict(self, pairs, batch_size: int = 32):
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            inputs = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            all_scores.extend(logits.squeeze(-1).tolist())
        return all_scores


def get_cross_encoder() -> _CrossEncoder:
    """Lazy-load the cross-encoder singleton."""
    global _model
    if _model is None:
        path = str(_LOCAL_PATH) if _LOCAL_PATH.exists() else _HF_MODEL
        logger.info(f"[CrossEncoder] Loading from {path} ...")
        _model = _CrossEncoder(path)
        logger.info("[CrossEncoder] Ready.")
    return _model


def warmup():
    """Call at server startup to pre-load the model (avoids cold-start on first query)."""
    get_cross_encoder()
