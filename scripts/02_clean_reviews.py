import json
import pandas as pd

BASE = "/home/Adithya/Desktop/Sem 7/BA/BAProject/topic1_app_review_dataset"

df = pd.read_csv(f"{BASE}/data/app_reviews_raw.csv")
df["review_date"] = pd.to_datetime(df["review_date"])
n_raw = len(df)

df["content"] = df["content"].fillna("").astype(str)
df = df.drop_duplicates(subset="review_id")
n_after_dedupe = len(df)

df = df[df["content"].str.strip().str.len() >= 3]
n_after_empty = len(df)


def ascii_ratio(s):
    return sum(c.isascii() for c in s) / max(len(s), 1)


df["ascii_ratio"] = df["content"].apply(ascii_ratio)
df = df[df["ascii_ratio"] >= 0.85]
n_after_lang = len(df)
df = df.drop(columns=["ascii_ratio"])

df["month"] = df["review_date"].dt.to_period("M").astype(str)
df["review_length"] = df["content"].str.split().str.len()

results = {
    "raw_rows": n_raw,
    "after_dedupe": n_after_dedupe,
    "after_empty_removed": n_after_empty,
    "after_non_english_removed": n_after_lang,
    "final_rows": len(df),
}

df.to_csv(f"{BASE}/data/app_reviews_clean.csv", index=False)
with open(f"{BASE}/data/cleaning_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Cleaning:", results)
print("Saved data/app_reviews_clean.csv (cleaned, not yet tagged) —", len(df), "rows")
