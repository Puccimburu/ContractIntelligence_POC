"""
Singleton cross-encoder for Phase 3 section re-ranking.

Loads from models/section_reranker/ (local, offline) if present,
otherwise falls back to HuggingFace Hub (requires internet).

Loading strategy (meta-device safe):
    1. from_pretrained() — on some transformers+accelerate combinations this
       initialises weights on the `meta` device rather than CPU.
    2. If meta tensors are detected, we materialise them on CPU by loading
       the weight file (safetensors preferred, then pytorch_model.bin) directly
       with torch, then assigning into a freshly constructed CPU model via
       load_state_dict(assign=True) — the approach recommended by PyTorch.
"""

import logging
import os
import pathlib

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_LOCAL_PATH = pathlib.Path("models/section_reranker")
_HF_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _load_state_dict_cpu(path: str) -> dict:
    """Load weights from safetensors or pytorch_model.bin, always onto CPU."""
    st_path = os.path.join(path, "model.safetensors")
    pt_path = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        return load_file(st_path, device="cpu")
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {path}")


class _CrossEncoder:
    """Minimal CrossEncoder wrapper — exposes .predict(pairs) matching sentence_transformers API."""

    def __init__(self, path: str, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Step 1: attempt normal load.
        model = AutoModelForSequenceClassification.from_pretrained(path)

        # Step 2: detect meta-device tensors.
        # When accelerate is installed, from_pretrained may put weights on the
        # meta device (a no-storage placeholder).  Calling .to("cpu") on a meta
        # tensor raises NotImplementedError; calling forward() raises:
        #   "Tensor on device cpu is not on the expected device meta!"
        # The correct fix (per PyTorch docs) is to_empty() + load_state_dict(assign=True).
        try:
            on_meta = any(p.device.type == "meta" for p in model.parameters())
        except Exception:
            on_meta = False

        if on_meta:
            logger.info("[CrossEncoder] Meta-device tensors detected — materialising on CPU")
            # Build a fresh, uninitialized CPU model (no meta tensors).
            config = AutoConfig.from_pretrained(path)
            cpu_model = AutoModelForSequenceClassification.from_config(config)
            cpu_model = cpu_model.to_empty(device="cpu")
            # Load actual weights from disk and assign into the empty model.
            sd = _load_state_dict_cpu(path)
            try:
                # assign=True requires PyTorch >= 2.1; replaces tensors in-place
                cpu_model.load_state_dict(sd, strict=True, assign=True)
            except TypeError:
                # Fallback for older PyTorch: copy_ each parameter manually
                own_sd = cpu_model.state_dict()
                for k in sd:
                    if k in own_sd:
                        own_sd[k].copy_(sd[k])
                cpu_model.load_state_dict(own_sd, strict=False)
            model = cpu_model
            logger.info("[CrossEncoder] CPU materialisation complete")

        self.model = model
        self.model.eval()

    def predict(self, pairs, batch_size: int = 32):
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            # Guard: replace empty strings with a single space so the tokenizer
            # never receives an empty sequence (causes "index out of range in self")
            queries = [p[0] if p[0] and p[0].strip() else " " for p in batch]
            passages = [p[1] if p[1] and p[1].strip() else " " for p in batch]
            inputs = self.tokenizer(
                queries,
                passages,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            scores = logits.squeeze(-1)
            # squeeze(-1) leaves shape [batch] when num_labels==1; if the model
            # returns multi-label logits take the first (relevance) column.
            if scores.dim() > 1:
                scores = scores[:, 0]
            all_scores.extend(scores.tolist())
        return all_scores


def get_cross_encoder() -> _CrossEncoder:
    """Lazy-load the cross-encoder singleton."""
    global _model
    if _model is None:
        path = str(_LOCAL_PATH) if _LOCAL_PATH.exists() else _HF_MODEL
        logger.info(f"[CrossEncoder] Loading from {path} ...")
        _model = _CrossEncoder(path)
        logger.info("[CrossEncoder] Ready (device=cpu).")
    return _model


def reset_cross_encoder() -> None:
    """Force a reload on the next get_cross_encoder() call. Useful after config changes."""
    global _model
    _model = None


def warmup():
    """Call at server startup to pre-load the model (avoids cold-start on first query)."""
    get_cross_encoder()
