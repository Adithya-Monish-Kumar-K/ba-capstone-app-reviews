"""
Stage 2: Cleaning (Person 1)

Per app: drop duplicate review IDs, empty/very short reviews (< 3 characters), reviews with no letters
at all (emoji- or punctuation-only) and non-English reviews (fewer than 85% of the letters are basic
Latin a-z). Language is judged on letters only, so emojis do not turn an English review like
"good 👍" into a "non-English" one. Then derive `month` and `review_length` (words).
Input  data/raw/<app>.csv.gz  ->  output data/clean/<app>.csv.gz, data/cleaning_results.json
"""
import json
from concurrent.futures import ProcessPoolExecutor

from apps import APPS
from data_io import DATA_DIR, read_app, write_app


def latin_letter_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def has_letters(s):
    return any(c.isalpha() for c in s)


def clean_app(slug):
    df = read_app("raw", slug, parse_dates=["review_date"])
    counts = {"raw_rows": len(df)}

    df["content"] = df["content"].fillna("").astype(str)
    df = df.drop_duplicates(subset="review_id")
    counts["after_dedupe"] = len(df)

    df = df[df["content"].str.strip().str.len() >= 3]
    counts["after_empty_removed"] = len(df)

    df = df[df["content"].map(has_letters)]
    counts["after_no_letters_removed"] = len(df)

    df = df[df["content"].map(latin_letter_ratio) >= 0.85]
    counts["after_non_english_removed"] = len(df)

    df = df.assign(
        month=df["review_date"].dt.to_period("M").astype(str),
        review_length=df["content"].str.split().str.len(),
    )
    counts["final_rows"] = len(df)
    counts["app_version_missing"] = int(df["app_version"].isna().sum())
    write_app(df, "clean", slug)
    return slug, counts


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=6) as pool:
        per_app = dict(pool.map(clean_app, APPS))

    keys = ["raw_rows", "after_dedupe", "after_empty_removed", "after_no_letters_removed", "after_non_english_removed",
            "final_rows", "app_version_missing"]
    total = {k: sum(c[k] for c in per_app.values()) for k in keys}
    results = {**total, "per_app": {APPS[s]["name"]: c for s, c in per_app.items()}}
    with open(DATA_DIR / "cleaning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Cleaning:", total)
    for slug, c in per_app.items():
        print(f"  {APPS[slug]['name']:10s} {c['raw_rows']:>8,} -> {c['final_rows']:>8,}")
    print(f"Saved data/clean/*.csv.gz (cleaned, not yet tagged) — {total['final_rows']:,} rows")
