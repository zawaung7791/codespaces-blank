"""
Download and combine public sarcasm datasets into a single reviews_sarcasm.csv
with columns: text, label, source.

Requires internet access (works inside the Codespace, not in a sandboxed
environment without network).

Datasets pulled:
  1. raquiba/Sarcasm_News_Headline   - Onion (sarcastic) vs HuffPost (real), ~28k, clean labels
  2. CreativeLang/SARC_Sarcasm       - large self-annotated Reddit corpus (sampled down)
  3. Dalilame/Multi-Sarcasm          - Reddit + chatbot conversational pairs, graded sarcasm

None of these are customer-review text specifically (there's no single clean
public review-sarcasm dataset), so treat this as a starting corpus to fine-tune
a general sarcasm detector, then adapt with real labeled reviews later for
best in-domain performance.

Usage:
    python download_data.py --out reviews_sarcasm.csv --sarc_sample 50000
"""

import argparse
import polars as pl
from datasets import load_dataset


def load_news_headlines() -> pl.DataFrame:
    ds = load_dataset("raquiba/Sarcasm_News_Headline", split="train")
    df = pl.from_pandas(ds.to_pandas())
    # Normalize column names
    text_col = "headline" if "headline" in df.columns else df.columns[0]
    label_col = "is_sarcastic" if "is_sarcastic" in df.columns else "label"
    df = df.select([
        pl.col(text_col).alias("text"),
        pl.col(label_col).cast(pl.Int64).alias("label"),
    ]).with_columns(pl.lit("news_headlines").alias("source"))
    return df


def load_sarc(sample_size: int) -> pl.DataFrame:
    ds = load_dataset("CreativeLang/SARC_Sarcasm", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= sample_size:
            break
        rows.append(row)
    df = pl.from_dicts(rows)
    text_col = "comment" if "comment" in df.columns else "text"
    label_col = "label" if "label" in df.columns else "is_sarcastic"
    df = df.select([
        pl.col(text_col).alias("text"),
        pl.col(label_col).cast(pl.Int64).alias("label"),
    ]).with_columns(pl.lit("sarc_reddit").alias("source"))
    return df


def load_multi_sarcasm() -> pl.DataFrame:
    ds = load_dataset("Dalilame/Multi-Sarcasm", split="train")
    df = pl.from_pandas(ds.to_pandas())
    df = df.select([
        pl.col("comment").alias("text"),
        pl.col("label").cast(pl.Int64).alias("label"),
    ]).with_columns(pl.lit("multi_sarcasm_reddit").alias("source"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reviews_sarcasm.csv")
    ap.add_argument("--sarc_sample", type=int, default=50000,
                     help="SARC is 1.3M+ rows; sample this many to keep things fast")
    args = ap.parse_args()

    frames = []

    print("Loading News Headlines dataset...")
    try:
        frames.append(load_news_headlines())
        print(f"  -> {len(frames[-1])} rows")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("Loading SARC (Reddit) dataset (streaming, sampled)...")
    try:
        frames.append(load_sarc(args.sarc_sample))
        print(f"  -> {len(frames[-1])} rows")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("Loading Multi-Sarcasm dataset...")
    try:
        frames.append(load_multi_sarcasm())
        print(f"  -> {len(frames[-1])} rows")
    except Exception as e:
        print(f"  FAILED: {e}")

    if not frames:
        raise RuntimeError("All dataset downloads failed - check your internet connection.")

    combined = pl.concat(frames, how="vertical_relaxed")
    combined = combined.drop_nulls(subset=["text", "label"])
    combined = combined.filter(pl.col("text").str.len_chars() > 0)
    combined = combined.unique(subset=["text"])

    print(f"\nCombined dataset: {len(combined)} rows")
    print(combined.group_by(["source", "label"]).len().sort(["source", "label"]))

    combined.write_csv(args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
