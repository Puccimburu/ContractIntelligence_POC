"""
Fine-tune the sentence transformer (section embedder) for legal contract retrieval.

Uses MultipleNegativesRankingLoss — the most effective loss for retrieval models.
Each row in your CSV is a (query, positive_section) pair. All other positives
in the same batch are treated as in-batch negatives automatically.

Usage
-----
    python -m scripts.train_section_embedder \
        --data_path data/retrieval_training_data.csv \
        --output_dir models/section_embedder \
        --base_model all-MiniLM-L6-v2 \
        --epochs 3 \
        --batch_size 32

Input CSV format
----------------
    query,positive
    "What is the notice period for termination?","Either party may terminate this Agreement by giving 30 days written notice..."
    "Who owns IP created under this agreement?","All intellectual property created by Supplier shall vest in Client..."

Tips for good training data
----------------------------
- 500–2000 pairs is a good starting point
- Queries should look like real user questions (not keyword searches)
- Positives should be the exact section text that answers the query
- Diversity matters more than volume — cover all clause types
- You can generate synthetic queries from your contracts using an LLM:
    "Write 3 questions that this contract section answers: {section_text}"

Output
------
    models/section_embedder/    <- fine-tuned model ready for use
"""

import argparse
import os
import pathlib
import sys

import pandas as pd


def load_data(data_path: str):
    """Load and validate the training CSV."""
    df = pd.read_csv(data_path)

    if "query" not in df.columns or "positive" not in df.columns:
        sys.exit(
            f"ERROR: CSV must have 'query' and 'positive' columns. "
            f"Found: {list(df.columns)}"
        )

    df["query"] = df["query"].fillna("").str.strip()
    df["positive"] = df["positive"].fillna("").str.strip()
    df = df[(df["query"] != "") & (df["positive"] != "")]

    print(f"\nLoaded {len(df)} (query, positive) pairs.")
    return df


def train(args):
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"  Section Embedder Fine-tuning")
    print(f"{'='*60}")
    print(f"  Base model  : {args.base_model}")
    print(f"  Data path   : {args.data_path}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"{'='*60}\n")

    df = load_data(args.data_path)

    # Train / validation split (90 / 10)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading base model {args.base_model} ...")
    model = SentenceTransformer(args.base_model)

    # ── Training data ─────────────────────────────────────────────────────────
    train_examples = [
        InputExample(texts=[row["query"], row["positive"]])
        for _, row in train_df.iterrows()
    ]

    # MultipleNegativesRankingLoss requires shuffle=True and no repeated queries
    # in the same batch. DataLoader handles this.
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,     # avoids single-sample batches which break the loss
    )

    loss = MultipleNegativesRankingLoss(model)

    # ── Evaluator (optional but useful) ──────────────────────────────────────
    # Build a small IR evaluator from the validation set so we can track
    # retrieval quality (MRR@10, NDCG@10) during training.
    queries = {str(i): row["query"] for i, row in val_df.iterrows()}
    corpus  = {str(i): row["positive"] for i, row in val_df.iterrows()}
    relevant_docs = {str(i): {str(i)} for i in range(len(val_df))}

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="val",
        show_progress_bar=False,
    )

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)

    print(f"\nStarting fine-tuning ({args.epochs} epoch(s), warmup={warmup_steps} steps) ...\n")
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\nDone. Fine-tuned model saved to: {output_dir.resolve()}")
    print(
        "\nIMPORTANT: Re-embed all existing contracts after fine-tuning.\n"
        "The new model produces different vectors — old Qdrant embeddings will\n"
        "give poor results until re-indexed. To re-embed:\n"
        "  1. Clear the 'contract_sections' Qdrant collection\n"
        "  2. Re-upload (or re-process) your contracts via the API\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the Legal Desk section embedder for contract retrieval."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to CSV with 'query' and 'positive' columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/section_embedder",
        help="Directory to save the fine-tuned model (default: models/section_embedder).",
    )
    parser.add_argument(
        "--base_model",
        default="all-MiniLM-L6-v2",
        help="Base sentence transformer to fine-tune (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32). Larger batches = more negatives = better quality.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        sys.exit(f"ERROR: Data file not found: {args.data_path}")

    train(args)


if __name__ == "__main__":
    main()
