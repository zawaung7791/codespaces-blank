"""
Baseline sarcasm detector: TF-IDF + Logistic Regression.

Usage:
    python baseline_sarcasm.py --data reviews.csv

Expects a CSV with columns:
    text   -> the review text
    label  -> 1 if sarcastic, 0 if not

If you don't have labeled data yet, this script can bootstrap weak labels
from star-rating / sentiment mismatch (see `weak_label_from_ratings` below)
so you have something to iterate on before hand-labeling.
"""

import argparse
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import joblib


def add_handcrafted_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add simple sarcasm-signal features to help debugging / feature importance."""
    return df.with_columns([
        pl.col("text").str.count_matches("!").alias("n_exclaim"),
        pl.col("text").str.count_matches(r"\.\.\.").alias("n_ellipsis"),
        pl.col("text").str.count_matches(r"\b[A-Z]{2,}\b").alias("n_caps_words"),
        pl.col("text").str.count_matches('"', literal=True).alias("n_quotes"),
    ])


def weak_label_from_ratings(df: pl.DataFrame, rating_col="rating", text_col="text") -> pl.DataFrame:
    """
    Optional bootstrap: flag high-rating reviews with negative-sounding text
    (or low-rating reviews with positive-sounding text) as candidate sarcasm.
    Requires a `rating` column (1-5). Use candidates as a starting point to
    hand-verify, not as ground truth.
    """
    from textblob import TextBlob  # pip install textblob

    def sentiment(t):
        return TextBlob(str(t)).sentiment.polarity  # -1..1

    df = df.with_columns(
        pl.col(text_col).map_elements(sentiment, return_dtype=pl.Float64).alias("sentiment")
    )
    df = df.with_columns(
        (
            ((pl.col(rating_col) >= 4) & (pl.col("sentiment") < -0.2))
            | ((pl.col(rating_col) <= 2) & (pl.col("sentiment") > 0.2))
        ).cast(pl.Int64).alias("candidate_sarcasm")
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with text,label columns")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out_model", default="sarcasm_baseline.joblib")
    args = ap.parse_args()

    df = pl.read_csv(args.data)
    df = df.drop_nulls(subset=[args.text_col, args.label_col])
    df = df.rename({args.text_col: "text"})
    df = add_handcrafted_features(df)

    texts = df["text"].to_list()
    labels = df[args.label_col].to_list()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=args.test_size, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handle imbalance
            C=1.0,
        )),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, preds, target_names=["not_sarcastic", "sarcastic"]))
    print(f"F1 (sarcastic class): {f1_score(y_test, preds):.3f}")

    # Inspect top features driving "sarcastic" predictions
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(clf.coef_[0])[-20:][::-1]
    print("\n=== Top features for 'sarcastic' ===")
    for i in top_idx:
        print(f"  {feature_names[i]:30s} weight={clf.coef_[0][i]:.3f}")

    joblib.dump(pipeline, args.out_model)
    print(f"\nModel saved to {args.out_model}")


if __name__ == "__main__":
    main()
