"""
Singleton sentence transformer for section embedding (Phase 2 RAG).

Loads from models/section_embedder/ (local, offline) if present,
otherwise falls back to HuggingFace Hub (requires internet).

Run  python -m scripts.download_sentence_transformer  once to cache locally.

Meta-device safety: when `accelerate` is installed, the internal HuggingFace
transformer may initialise weights on the `meta` device even when
device="cpu" is passed to SentenceTransformer.  After loading we detect
and materialise any meta tensors using the same to_empty+load_state_dict
approach used in cross_encoder_instance.py.
"""

import logging
import os
import pathlib

logger = logging.getLogger(__name__)

_LOCAL_PATH = pathlib.Path("models/section_embedder")
_HF_MODEL = "all-MiniLM-L6-v2"

_model = None


def _fix_meta_tensors(st_model):
    """
    Walk every sentence_transformers module that wraps a HuggingFace
    transformer (i.e. has an .auto_model attribute) and materialise any
    meta-device tensors onto CPU.

    This mirrors the pattern in cross_encoder_instance.py but applied to
    the internal transformer(s) inside a SentenceTransformer object.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    for module in st_model.modules():
        auto_model = getattr(module, "auto_model", None)
        if auto_model is None:
            continue

        try:
            on_meta = any(p.device.type == "meta" for p in auto_model.parameters())
        except Exception:
            on_meta = False

        if not on_meta:
            continue

        logger.info(
            "[SentenceTransformer] Meta-device tensors detected in %s — materialising on CPU",
            type(auto_model).__name__,
        )

        # Locate the weight file for this sub-model.
        # SentenceTransformer modules store their own path in module._model_card_vars
        # or we can infer it from the parent SentenceTransformer path.
        model_path = getattr(module, "_model_path", None)
        if model_path is None:
            # Fallback: use the SentenceTransformer root path + "0_Transformer"
            root = getattr(st_model, "_model_path", None) or str(_LOCAL_PATH)
            model_path = os.path.join(root, "0_Transformer")
            if not os.path.isdir(model_path):
                model_path = root

        st_file = os.path.join(model_path, "model.safetensors")
        pt_file = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(st_file):
            from safetensors.torch import load_file
            sd = load_file(st_file, device="cpu")
        elif os.path.exists(pt_file):
            sd = torch.load(pt_file, map_location="cpu")
        else:
            logger.warning(
                "[SentenceTransformer] No weight file found in %s — cannot fix meta tensors",
                model_path,
            )
            continue

        config = AutoConfig.from_pretrained(model_path)
        cpu_model = AutoModel.from_config(config)
        cpu_model = cpu_model.to_empty(device="cpu")

        try:
            cpu_model.load_state_dict(sd, strict=True, assign=True)
        except TypeError:
            own_sd = cpu_model.state_dict()
            for k in sd:
                if k in own_sd:
                    own_sd[k].copy_(sd[k])
            cpu_model.load_state_dict(own_sd, strict=False)

        module.auto_model = cpu_model
        logger.info("[SentenceTransformer] CPU materialisation complete for %s", type(auto_model).__name__)


def get_sentence_transformer():
    """Lazy-load the sentence transformer singleton."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        path = str(_LOCAL_PATH) if _LOCAL_PATH.exists() else _HF_MODEL
        logger.info(f"[SentenceTransformer] Loading from {path} ...")
        # Force device="cpu" to prevent accelerate from placing tensors on the
        # meta device, which causes "Cannot copy out of meta tensor" at encode time.
        _model = SentenceTransformer(path, device="cpu")
        # Belt-and-suspenders: fix any meta tensors that slipped through anyway.
        _fix_meta_tensors(_model)
        logger.info("[SentenceTransformer] Ready.")
    return _model
