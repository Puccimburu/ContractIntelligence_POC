"""
Run this once to download the cross-encoder model to models/section_reranker/
so the server never hits HuggingFace on startup.

Usage:
    python -m scripts.download_cross_encoder
"""

import pathlib
from sentence_transformers import CrossEncoder

OUTPUT_DIR = pathlib.Path("models/section_reranker")
HF_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if __name__ == "__main__":
    print(f"Downloading {HF_MODEL} ...")
    model = CrossEncoder(HF_MODEL, max_length=512)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"Saved to {OUTPUT_DIR.resolve()}")
