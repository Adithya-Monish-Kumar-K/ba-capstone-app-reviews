"""
Predict whether new, unseen reviews are "problematic" (would be rated 1-2 stars) with the saved models.

For each review text it repeats the exact steps used for training:
  1. issue tags            scripts/03_tag_issues.py (keyword rules, 9 flags)
  2. VADER sentiment       scripts/04_sentiment_analysis.py (neg / neu / pos / compound)
  3. review length         number of words
  4. TF-IDF -> SVD         models/tfidf_vectorizer.pkl, models/svd_model.pkl (200 text components)
  5. scaling               models/structural_scaler.pkl (the 15 structural features)
  6. classifier            models/logistic_regression.pkl and models/random_forest.pkl, flag when P >= threshold

The model predicts a low star rating from the review text, not a confirmed software failure.
`thumbs_up` is unknown for a brand-new review, so it defaults to 0.

Usage (from the repository root):
  python scripts/14_predict_review.py "Refund not received, customer care never replies"
  python scripts/14_predict_review.py "nice app" "app keeps crashing after update" --threshold 0.6
  python scripts/14_predict_review.py --file reviews.txt        # one review per line
"""
import argparse
import importlib
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tagger = importlib.import_module("03_tag_issues")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "models")

ISSUE_COLS = [f"issue_{c}" for c in tagger.TAXONOMY]      # same order as scripts/05_feature_engineering.py
SENTIMENT_COLS = ["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound"]
STRUCTURAL_COLS = ISSUE_COLS + SENTIMENT_COLS + ["review_length", "thumbs_up"]


def load_models():
    names = ["tfidf_vectorizer", "svd_model", "structural_scaler", "logistic_regression", "random_forest"]
    return {n: joblib.load(os.path.join(MODEL_DIR, f"{n}.pkl")) for n in names}


def vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        sys.exit("VADER lexicon missing. Run:  python -c \"import nltk; nltk.download('vader_lexicon')\"")


def features(texts, sia, models, thumbs_up=0):
    """Build the 215-feature matrix for a list of review texts (same steps as training)."""
    rows, tag_lists = [], []
    for t in texts:
        t = str(t) if t is not None else ""
        tags = tagger.tag_review_text(t)
        s = sia.polarity_scores(t)
        rows.append({**{c: tags[c] for c in ISSUE_COLS},
                     "sentiment_neg": s["neg"], "sentiment_neu": s["neu"], "sentiment_pos": s["pos"],
                     "sentiment_compound": s["compound"], "review_length": len(t.split()), "thumbs_up": thumbs_up})
        tag_lists.append([c[len("issue_"):] for c in ISSUE_COLS if tags[c]])
    structural = pd.DataFrame(rows)[STRUCTURAL_COLS].astype(float)
    text_part = models["svd_model"].transform(models["tfidf_vectorizer"].transform([str(t) for t in texts]))
    X = np.hstack([text_part, models["structural_scaler"].transform(structural)]).astype(np.float32)
    return X, structural, tag_lists


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reviews", nargs="*", help="review text(s) to classify")
    ap.add_argument("--file", help="text file with one review per line")
    ap.add_argument("--threshold", type=float, default=0.5, help="flag a review when P(problematic) >= this (default 0.5)")
    ap.add_argument("--thumbs-up", type=int, default=0, help="upvotes, if known (default 0, as for a new review)")
    args = ap.parse_args()

    texts = list(args.reviews)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        ap.error("give at least one review text or --file")

    warnings.filterwarnings("ignore")
    models = load_models()
    X, structural, tags = features(texts, vader(), models, args.thumbs_up)
    p_lr = models["logistic_regression"].predict_proba(X)[:, 1]
    p_rf = models["random_forest"].predict_proba(X)[:, 1]

    for i, t in enumerate(texts):
        s = structural.iloc[i]
        print(f"\nReview: {t}")
        print(f"  issue tags : {', '.join(tags[i]) or 'none'}")
        print(f"  sentiment  : compound {s['sentiment_compound']:+.2f} (neg {s['sentiment_neg']:.2f}, "
              f"neu {s['sentiment_neu']:.2f}, pos {s['sentiment_pos']:.2f}); {int(s['review_length'])} words")
        for name, p in (("Logistic Regression", p_lr[i]), ("Random Forest", p_rf[i])):
            print(f"  {name:19s}: P(problematic) = {p:.2f} -> {'PROBLEMATIC' if p >= args.threshold else 'not problematic'}")


if __name__ == "__main__":
    main()
