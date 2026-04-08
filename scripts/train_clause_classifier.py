"""
Train the BERT clause classifier for Legal Desk Phase 1 RAG.

Fine-tunes bert-base-uncased (or any HuggingFace model) on a CSV of labelled
contract sections, then saves the model to models/clause_classifier/.

Usage
-----
    python -m scripts.train_clause_classifier \
        --data_path data/clause_training_data.csv \
        --output_dir models/clause_classifier \
        --base_model bert-base-uncased \
        --epochs 5 \
        --batch_size 16 \
        --max_length 256

Input CSV format
----------------
    text,label
    "This Agreement shall commence on the Effective Date...",term_and_termination
    "All Confidential Information disclosed by either Party...",confidentiality

Supported labels (20 total)
----------------------------
    definitions, order_of_precedence, term_and_termination,
    limitation_of_liability, confidentiality, intellectual_property,
    governing_law, dispute_resolution, payment_terms, data_protection,
    warranties, indemnification, force_majeure, notices,
    schedule_or_appendix, business_continuity, audit_rights,
    subcontracting, general, other

Output
------
    models/clause_classifier/                <- final model (load directly)
    models/clause_classifier/checkpoint-N/  <- best checkpoint during training
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd

CLAUSE_TYPES = [
    "definitions", "order_of_precedence", "term_and_termination",
    "limitation_of_liability", "confidentiality", "intellectual_property",
    "governing_law", "dispute_resolution", "payment_terms", "data_protection",
    "warranties", "indemnification", "force_majeure", "notices",
    "schedule_or_appendix", "business_continuity", "audit_rights",
    "subcontracting", "general", "other",
]

LABEL2ID = {label: i for i, label in enumerate(CLAUSE_TYPES)}
ID2LABEL = {i: label for i, label in enumerate(CLAUSE_TYPES)}


def load_data(data_path: str):
    """Load and validate the training CSV."""
    df = pd.read_csv(data_path)

    if "text" not in df.columns or "label" not in df.columns:
        sys.exit(
            f"ERROR: CSV must have 'text' and 'label' columns. "
            f"Found: {list(df.columns)}"
        )

    # Normalise whitespace and lowercase labels
    df["text"] = df["text"].fillna("").str.strip()
    df["label"] = df["label"].str.strip().str.lower()

    # Report unknown labels
    unknown = set(df["label"].unique()) - set(CLAUSE_TYPES)
    if unknown:
        print(f"WARNING: Unknown labels will be mapped to 'other': {unknown}")
        df.loc[df["label"].isin(unknown), "label"] = "other"

    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"\nLoaded {len(df)} examples across {df['label'].nunique()} classes.")
    print(df["label"].value_counts().to_string())
    return df


def build_dataset(df, tokenizer, max_length: int):
    """Tokenise the dataframe and return a HuggingFace Dataset."""
    import torch
    from torch.utils.data import Dataset

    class ClauseDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    return ClauseDataset(df["text"].tolist(), df["label_id"].tolist())


def compute_metrics(eval_pred):
    """Accuracy metric for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}


def train(args):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    print(f"\n{'='*60}")
    print(f"  Clause Classifier Training")
    print(f"{'='*60}")
    print(f"  Base model  : {args.base_model}")
    print(f"  Data path   : {args.data_path}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max length  : {args.max_length}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    df = load_data(args.data_path)

    # Train / validation split (90 / 10)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["label_id"]
    )
    print(f"\nTrain: {len(train_df)} | Validation: {len(val_df)}")

    # ── Tokeniser & model ────────────────────────────────────────────────────
    print(f"\nLoading tokeniser from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading model from {args.base_model} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(CLAUSE_TYPES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = build_dataset(train_df, tokenizer, args.max_length)
    val_dataset = build_dataset(val_df, tokenizer, args.max_length)

    # ── Training arguments ───────────────────────────────────────────────────
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,           # keep only the 2 best checkpoints
        report_to="none",             # disable wandb / tensorboard
        fp16=False,                   # set True if you have a CUDA GPU
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training ...\n")
    trainer.train()

    # ── Save final model ─────────────────────────────────────────────────────
    print(f"\nSaving final model to {output_dir} ...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\nEvaluation on validation set:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nDone. Model saved to: {output_dir.resolve()}")
    print(
        "Restart the backend — it will auto-detect the new model and load it at startup."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the Legal Desk clause classifier."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to training CSV with 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/clause_classifier",
        help="Directory to save the trained model (default: models/clause_classifier).",
    )
    parser.add_argument(
        "--base_model",
        default="bert-base-uncased",
        help="HuggingFace base model to fine-tune (default: bert-base-uncased).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size per device (default: 16). Reduce to 8 if OOM.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length per input (default: 256).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        sys.exit(f"ERROR: Data file not found: {args.data_path}")

    train(args)


if __name__ == "__main__":
    main()
