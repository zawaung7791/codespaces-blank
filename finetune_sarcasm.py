"""
Fine-tune a transformer (default: distilroberta-base) for sarcasm detection.

Install:
    pip install transformers datasets scikit-learn torch accelerate polars --break-system-packages

Usage:
    python finetune_sarcasm.py --data reviews.csv --epochs 3

Expects a CSV with columns:
    text   -> the review text
    label  -> 1 if sarcastic, 0 if not

Swap --model_name for something stronger later, e.g.:
    roberta-base, microsoft/deberta-v3-base (bigger, slower, usually better)
"""

import argparse
import numpy as np
import polars as pl
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--model_name", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--out_dir", default="./sarcasm_model")
    args = ap.parse_args()

    df = pl.read_csv(args.data)
    df = df.drop_nulls(subset=[args.text_col, args.label_col])
    df = df.rename({args.text_col: "text", args.label_col: "label"})
    df = df.with_columns(pl.col("label").cast(pl.Int64))

    texts = df["text"].to_list()
    labels = df["label"].to_list()
    indices = list(range(len(texts)))

    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=labels
    )

    train_ds = Dataset.from_dict({
        "text": [texts[i] for i in train_idx],
        "label": [labels[i] for i in train_idx],
    })
    val_ds = Dataset.from_dict({
        "text": [texts[i] for i in val_idx],
        "label": [labels[i] for i in val_idx],
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Final eval ===")
    print(trainer.evaluate())

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"\nModel saved to {args.out_dir}")


if __name__ == "__main__":
    main()
