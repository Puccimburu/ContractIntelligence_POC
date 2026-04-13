"""
Singleton clause classifier — loaded once at startup, reused per request.

Mirrors the pattern in sentence_transformer_instance.py.

Usage:
    from src.utils.clause_classifier_instance import classify_sections_batch

    sections = [{"sectionTitle": "...", "content": "...", ...}, ...]
    sections = classify_sections_batch(sections)
    # each section now has sec["clauseType"] set

Falls back silently to "other" if the model directory does not exist yet
(i.e. before the first training run).  This keeps the pipeline functional
during development.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

_model      = None
_tokenizer  = None
_available  = None   # None = not yet checked; True/False after first call

_BASE_DIR = "models/clause_classifier"

def _resolve_model_path():
    """Return the deepest checkpoint dir if the base dir only has checkpoints."""
    import pathlib
    base = pathlib.Path(_BASE_DIR)
    if not base.exists():
        return _BASE_DIR
    # If there's a config.json directly, use base
    if (base / "config.json").exists():
        return str(base)
    # Otherwise look for a checkpoint-* subdirectory
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if checkpoints:
        return str(checkpoints[-1])
    return str(base)

MODEL_PATH = _resolve_model_path()
MAX_LENGTH = 256

CLAUSE_TYPES = [
    "definitions", "order_of_precedence", "term_and_termination",
    "limitation_of_liability", "confidentiality", "intellectual_property",
    "governing_law", "dispute_resolution", "payment_terms", "data_protection",
    "warranties", "indemnification", "force_majeure", "notices",
    "schedule_or_appendix", "business_continuity", "audit_rights",
    "subcontracting", "general", "other",
]


def _load_model():
    """Load model + tokenizer from MODEL_PATH. Sets _available flag."""
    global _model, _tokenizer, _available
    import pathlib
    if not pathlib.Path(MODEL_PATH).exists():
        logger.warning(
            "[ClauseClassifier] Model directory '%s' not found. "
            "Run scripts/train_clause_classifier.py to build it. "
            "Falling back to LLM classification until then.",
            MODEL_PATH,
        )
        _available = False
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("[ClauseClassifier] Loading model from %s …", MODEL_PATH)
        # Checkpoint dirs don't include tokenizer files — fall back to the
        # base BERT vocab which is identical for any bert-base-uncased derivative.
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        # Detect and fix meta-device tensors (same issue as cross-encoder).
        on_meta = any(p.device.type == "meta" for p in _model.parameters())
        if on_meta:
            import os as _os
            import pathlib as _pathlib
            logger.info("[ClauseClassifier] Meta-device tensors detected — materialising on CPU")
            mp = _pathlib.Path(MODEL_PATH)
            st = mp / "model.safetensors"
            pt = mp / "pytorch_model.bin"
            if st.exists():
                from safetensors.torch import load_file as _lf
                sd = _lf(str(st), device="cpu")
            elif pt.exists():
                sd = torch.load(str(pt), map_location="cpu")
            else:
                raise FileNotFoundError(f"No weight file in {MODEL_PATH}")
            from transformers import AutoConfig as _AC
            cfg = _AC.from_pretrained(MODEL_PATH)
            cpu_m = AutoModelForSequenceClassification.from_config(cfg)
            cpu_m = cpu_m.to_empty(device="cpu")
            try:
                cpu_m.load_state_dict(sd, strict=True, assign=True)
            except TypeError:
                own = cpu_m.state_dict()
                for k in sd:
                    if k in own:
                        own[k].copy_(sd[k])
                cpu_m.load_state_dict(own, strict=False)
            _model = cpu_m

        _model.eval()
        _available = True
        logger.info("[ClauseClassifier] Model loaded (%d labels).", _model.config.num_labels)
    except Exception as e:
        logger.error("[ClauseClassifier] Failed to load model: %s — will use LLM fallback.", e)
        _available = False


def is_available() -> bool:
    """Return True if the fine-tuned classifier is loaded and ready."""
    global _available
    if _available is None:
        _load_model()
    return bool(_available)


def classify_sections_batch(sections: List[dict]) -> List[dict]:
    """
    Run clauseType inference on a list of section dicts in place.

    Sets sec["clauseType"] on every section.
    If the model is not available, sections are returned unchanged
    (the caller's LLM fallback will handle classification).

    Args:
        sections: list of section dicts with at least 'sectionTitle' and 'content' keys.

    Returns:
        The same list with 'clauseType' populated.
    """
    if not sections:
        return sections

    if not is_available():
        return sections   # LLM fallback in _classify_clause_types() takes over

    import torch

    try:
        texts = [
            f"{s.get('sectionTitle', '')} [SEP] {(s.get('content') or '')[:300]}"
            for s in sections
        ]

        # Run in sub-batches of 32 to avoid OOM on long documents
        all_predicted_ids: List[int] = []
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i + 32]
            inputs = _tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = _model(**inputs).logits
            all_predicted_ids.extend(logits.argmax(dim=-1).tolist())

        id2label = _model.config.id2label
        for sec, pid in zip(sections, all_predicted_ids):
            sec["clauseType"] = id2label.get(pid, "other")

        logger.debug(
            "[ClauseClassifier] Classified %d section(s) via fine-tuned model.",
            len(sections),
        )

    except Exception as e:
        logger.warning(
            "[ClauseClassifier] Batch inference failed: %s — clauseType left unset for LLM fallback.",
            e,
        )

    return sections


def warmup():
    """
    Pre-load the model at startup so the first file processed has no cold-start delay.
    Call from main.py lifespan alongside ensure_section_collection().
    """
    is_available()   # triggers _load_model() if not yet checked
