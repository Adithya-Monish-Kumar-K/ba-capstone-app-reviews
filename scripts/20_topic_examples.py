"""
Stage 3 of Review 2, Method 1 (Text Mining): example reviews per topic for the dashboard (#139).

`data/textmining/topic_exemplars.csv` holds the cleaned, lemmatised text of the best-fitting reviews, which is
hard to read. This script picks readable examples instead: for every complaint topic and every app in its domain,
the reviews that fit the topic best (highest topic confidence) and are a sensible length, with their original text.

Input:  data/textmining/review_topics.csv.gz (from 17_topic_modelling.py), data/tagged/*.csv.gz (original text)
Output: data/textmining/topic_examples.csv
          domain, topic_id, topic_label, app_name, rank, review_id, review_date, score, sentiment_compound,
          topic_confidence, content

Selection: 1-2 star reviews of 60-300 characters, at least 95% ASCII letters/punctuation (the topic models are
English), topic confidence of at least 0.5, no duplicate text; the top EXAMPLES_PER_CELL by confidence per topic and app.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import read_stage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TM_DIR = REPO_ROOT / "data" / "textmining"
EXAMPLES_PER_CELL = 3
MIN_CHARS, MAX_CHARS = 60, 300
MIN_CONFIDENCE = 0.5


def mostly_ascii(text):
    return sum(ch.isascii() for ch in text) / max(len(text), 1) >= 0.95


def main():
    topics = pd.read_csv(TM_DIR / "review_topics.csv.gz")
    print(f"{len(topics):,} complaint reviews with a topic")
    reviews = read_stage("tagged", columns=["review_id", "content", "review_date"])
    reviews["review_id"] = reviews["review_id"].astype(str)
    df = topics.merge(reviews, on="review_id", how="inner")
    df["content"] = df["content"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

    ok = (df["content"].str.len().between(MIN_CHARS, MAX_CHARS) & (df["topic_confidence"] >= MIN_CONFIDENCE)
          & df["content"].map(mostly_ascii))
    df = df[ok].drop_duplicates(["app_name", "content"])
    df = df.sort_values(["topic_confidence", "review_id"], ascending=[False, True])
    keys = ["domain", "dominant_topic_id", "app_name"]
    df["rank"] = df.groupby(keys).cumcount() + 1
    out = df[df["rank"] <= EXAMPLES_PER_CELL].rename(columns={"dominant_topic_id": "topic_id", "dominant_topic_label": "topic_label"})
    out = out[["domain", "topic_id", "topic_label", "app_name", "rank", "review_id", "review_date", "score",
               "sentiment_compound", "topic_confidence", "content"]].sort_values(["domain", "topic_id", "app_name", "rank"])
    out["topic_confidence"] = out["topic_confidence"].round(3)
    out.to_csv(TM_DIR / "topic_examples.csv", index=False)
    cells = out.groupby(["domain", "topic_id", "app_name"]).ngroups
    print(f"Saved {len(out)} examples for {cells} topic-app cells to {(TM_DIR / 'topic_examples.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
