"""
Run this once to download the sentence transformer model to models/section_embedder/
so the server never hits HuggingFace on startup.

Usage:
    python -m scripts.download_sentence_transformer
"""

import pathlib
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = pathlib.Path("models/section_embedder")
HF_MODEL = "all-MiniLM-L6-v2"

if __name__ == "__main__":
    print(f"Downloading {HF_MODEL} ...")
    model = SentenceTransformer(HF_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"Saved to {OUTPUT_DIR.resolve()}")
