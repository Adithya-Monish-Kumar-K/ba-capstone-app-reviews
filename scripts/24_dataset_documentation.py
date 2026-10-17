"""Data dictionary, sample dataset and validation summary for the final dataset.

The final validated dataset is data/tagged/<app>.csv.gz (11 files, one per app). This script
checks it again, end to end, and writes:

- DATA_DICTIONARY.md                       every column: type, range or values, missing share, meaning
- data/sample/final_dataset_sample.csv     10 random reviews per app (110 rows, all 29 columns)
- DATA_VALIDATION.md                       each check, its result, and the counts from raw to final
- data/dataset_validation.json             the same numbers, machine-readable

Run: python scripts/24_dataset_documentation.py   (about 1 minute)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from apps import APP_DOMAIN, APPS  # noqa: E402

START, END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-21")  # end is exclusive
ISSUES = ["crash_bugs_stability", "payment_refund", "delivery_delay", "order_quality_fulfillment",
          "cancellation_return", "customer_support", "account_login_otp", "pricing_charges_fraud", "ui_ux_update"]
ISSUE_COLS = [f"issue_{c}" for c in ISSUES]

MEANING = {
    "app_id": ("Play Store package name", "collected"),
    "app_name": ("One of the 11 apps", "collected"),
    "domain": ("Food & Grocery, Shopping or Payments", "collected"),
    "review_id": ("Play Store's unique review id", "collected"),
    "score": ("Star rating given by the user (1–5)", "collected"),
    "content": ("Review text", "collected"),
    "thumbs_up": ("Users who marked the review helpful", "collected"),
    "app_version": ("App version on the reviewer's phone; empty when the Play Store gives none", "collected"),
    "review_date": ("Date and time the review was posted", "collected"),
    "month": ("Year-month of review_date", "derived in cleaning"),
    "review_length": ("Number of words in content", "derived in cleaning"),
    "issue_count": ("Number of the 9 issue tags the review matches", "derived in tagging"),
    "has_issue": ("1 if issue_count > 0", "derived in tagging"),
    "primary_issue": ("Tag with the most keyword matches, or 'none'", "derived in tagging"),
    "all_issues": ("All matched tags joined by ';', or 'none'", "derived in tagging"),
    "sentiment_neg": ("VADER share of negative words (0–1)", "derived in sentiment"),
    "sentiment_neu": ("VADER share of neutral words (0–1)", "derived in sentiment"),
    "sentiment_pos": ("VADER share of positive words (0–1)", "derived in sentiment"),
    "sentiment_compound": ("VADER overall score from −1 (very negative) to +1 (very positive)", "derived in sentiment"),
    "sentiment_label": ("Positive if compound ≥ 0.05, Negative if ≤ −0.05, else Neutral", "derived in sentiment"),
}
ISSUE_NAMES = ["Crash & Stability", "Payment & Refund", "Delivery Delay", "Order Quality & Fulfilment",
               "Cancellation & Return", "Customer Support", "Account / Login / OTP", "Pricing, Charges & Fraud",
               "UI/UX & Update"]
for c, name in zip(ISSUES, ISSUE_NAMES):
    MEANING[f"issue_{c}"] = (f"1 if the text matches the {name} keywords", "derived in tagging")


def load(stage):
    return pd.concat([pd.read_csv(ROOT / "data" / stage / f"{slug}.csv.gz", dtype={"app_version": str})
                      for slug in APPS], ignore_index=True)


def main():
    raw, clean, df = load("raw"), load("clean"), load("tagged")
    df["review_date"] = pd.to_datetime(df["review_date"])
    log = json.load(open(ROOT / "data" / "cleaning_results.json"))

    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    n = len(df)
    check("Row count matches the cleaning log", n == log["final_rows"] == len(clean),
          f"{n:,} tagged rows, {len(clean):,} cleaned rows, {log['final_rows']:,} in cleaning_results.json")
    check("Same reviews in the cleaned and tagged files",
          set(df.review_id) == set(clean.review_id), "review_id sets are identical")
    check("No duplicate reviews", df.review_id.is_unique, f"{df.review_id.duplicated().sum()} duplicate review_id")
    req = [c for c in df.columns if c != "app_version"]
    miss = df[req].isna().sum()
    check("No missing values outside app_version", miss.sum() == 0,
          "0 missing in the other 28 columns" if miss.sum() == 0 else str(miss[miss > 0].to_dict()))
    av = df.app_version.isna().sum()
    check("app_version missing share is as documented", av == log["app_version_missing"],
          f"{av:,} missing ({av / n:.1%}); kept, not used as a model input")
    check("Ratings are 1–5", df.score.between(1, 5).all(), f"min {df.score.min()}, max {df.score.max()}")
    check("Helpful votes are not negative", (df.thumbs_up >= 0).all(), f"min {df.thumbs_up.min()}, max {df.thumbs_up.max():,}")
    check("Every review is inside 1 Apr – 20 Sep 2026",
          df.review_date.between(START, END, inclusive="left").all(),
          f"{df.review_date.min():%d %b %Y %H:%M} to {df.review_date.max():%d %b %Y %H:%M}")
    check("month agrees with review_date", (df.month == df.review_date.dt.strftime("%Y-%m")).all(), "all rows")
    wl = df.content.astype(str).str.split().str.len()
    check("review_length is the word count of content", (wl == df.review_length).all(), "all rows")
    check("Every review has letters", df.content.astype(str).str.contains(r"[A-Za-z]").all(),
          "emoji-only and punctuation-only reviews were removed in cleaning")
    check("Domain matches the app", (df.domain == df.app_name.map(APP_DOMAIN)).all(), "per scripts/apps.py")
    check("issue_count equals the sum of the 9 issue flags", (df[ISSUE_COLS].sum(axis=1) == df.issue_count).all(), "all rows")
    check("has_issue agrees with issue_count", (df.has_issue == (df.issue_count > 0).astype(int)).all(), "all rows")
    check("primary_issue is 'none' exactly when no tag matched",
          ((df.primary_issue == "none") == (df.issue_count == 0)).all(), "all rows")
    sums = df[["sentiment_neg", "sentiment_neu", "sentiment_pos"]].sum(axis=1)
    # VADER skips one-letter tokens, so a review made only of them ("o k", "o s m") scores 0 on all four
    zero = (sums == 0) & (df.sentiment_compound == 0)
    check("VADER scores are in range and neg + neu + pos ≈ 1",
          df.sentiment_compound.between(-1, 1).all() and ((sums.sub(1).abs() <= 0.01) | zero).all(),
          f"compound {df.sentiment_compound.min():.3f} to {df.sentiment_compound.max():.3f}; sums within 0.01 of 1, "
          f"except {zero.sum()} reviews made only of one-letter tokens (\"o k\"), which VADER scores 0 on all four (Neutral)")
    lab = np.where(df.sentiment_compound >= 0.05, "Positive", np.where(df.sentiment_compound <= -0.05, "Negative", "Neutral"))
    check("sentiment_label follows the compound thresholds", (lab == df.sentiment_label).all(), "±0.05")
    days = df.groupby("app_name").review_date.apply(lambda s: s.dt.date.nunique())
    check("Daily coverage", (days.drop("Zomato") == 173).all() and days["Zomato"] == 172,
          "173 of 173 days for 10 apps; Zomato 172 (24 Jul missing in the Play Store itself)")

    # raw-to-final counts and suitability for each unit
    steps = [("Raw reviews collected", log["raw_rows"]), ("Duplicates removed", log["raw_rows"] - log["after_dedupe"]),
             ("Empty or under 3 characters removed", log["after_dedupe"] - log["after_empty_removed"]),
             ("No letters (emoji or punctuation only) removed", log["after_empty_removed"] - log["after_no_letters_removed"]),
             ("Non-English removed", log["after_no_letters_removed"] - log["after_non_english_removed"]),
             ("Final validated reviews", log["final_rows"])]
    neg = int((df.score <= 2).sum())
    pos = int((df.score >= 4).sum())
    tm = json.loads((ROOT / "data" / "textmining_summary.json").read_text())["selection"]   # from scripts/16
    units = {
        "Unit 1 (EDA)": f"all {n:,} reviews, 29 columns, 11 apps, 173 days",
        "Unit 2 (classifier)": f"target is_problematic = score ≤ 2: {neg:,} positives ({neg / n:.1%}); "
                               "inputs: content, 9 issue flags, 4 VADER scores, review_length, thumbs_up",
        "Unit 3, text mining": f"content of the {neg:,} 1–2★ reviews ({tm['complaint']['kept']:,} with ≥ 3 content words) and "
                               f"{pos:,} 4–5★ reviews as a contrast",
        "Unit 3, time series": "review_date + score: daily 1–2★ counts for 11 apps × 173 days; "
                               f"app_version for release events ({df.groupby(['app_name', 'app_version']).ngroups:,} app versions, 224 new releases)",
    }

    (ROOT / "data" / "sample").mkdir(exist_ok=True)
    sample = df.groupby("app_name", group_keys=False).sample(10, random_state=42)
    sample = sample.sort_values(["domain", "app_name", "review_date"])
    sample.assign(review_date=sample.review_date.dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(
        ROOT / "data" / "sample" / "final_dataset_sample.csv", index=False)

    out = {"final_dataset": "data/tagged/<app>.csv.gz (11 files)", "rows": n, "columns": df.shape[1],
           "checks": checks, "all_checks_passed": all(c["passed"] for c in checks),
           "raw_to_final": dict(steps), "per_app_rows": df.app_name.value_counts().to_dict(),
           "unit_suitability": units}
    json.dump(out, open(ROOT / "data" / "dataset_validation.json", "w"), indent=2, ensure_ascii=False)

    # data dictionary
    lines = ["# Data Dictionary",
             "",
             f"Final validated dataset: `data/tagged/<app>.csv.gz`, one gzipped CSV per app, {n:,} rows and "
             f"{df.shape[1]} columns in total. Collection is described in `DATA_SOURCES.md`, the checks in "
             "`DATA_VALIDATION.md`, and a 110-row sample is in `data/sample/final_dataset_sample.csv`.",
             "",
             "Generated by `scripts/24_dataset_documentation.py`.",
             "",
             "| # | Column | Type | Values / range | Missing | Meaning | Source |",
             "|---:|---|---|---|---:|---|---|"]
    for i, c in enumerate(df.columns, 1):
        s = df[c]
        if c == "review_date":
            typ, rng = "datetime", f"{s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
        elif pd.api.types.is_integer_dtype(s):
            typ = "integer (0/1)" if set(s.unique()) <= {0, 1} else "integer"
            rng = f"{s.min():,} to {s.max():,}"
        elif pd.api.types.is_float_dtype(s):
            typ, rng = "float", f"{s.min():.3f} to {s.max():.3f}"
        else:
            u = s.dropna().nunique()
            typ = "text" if c in ("content", "all_issues", "review_id") else "category"
            rng = f"{u:,} distinct" if u > 12 else ", ".join(sorted(map(str, s.dropna().unique())))
        m = s.isna().mean()
        meaning, source = MEANING[c]
        lines.append(f"| {i} | `{c}` | {typ} | {rng} | {m:.1%} | {meaning} | {source} |")
    lines += ["",
              "**Target used in Unit 2:** `is_problematic = 1` when `score` is 1 or 2 "
              f"({neg:,} reviews, {neg / n:.1%}); it is computed from `score` when the models are trained, so it is "
              "not stored as a column.",
              "",
              "**Derived for Unit 3 (not columns of this file):** daily and weekly 1–2★ counts per app "
              "(`data/timeseries/daily_app.csv`), release events from `app_version` "
              "(`data/timeseries/release_events.csv`), and the cleaned complaint and praise corpora "
              "(`data/textmining/corpus.csv.gz`)."]
    (ROOT / "DATA_DICTIONARY.md").write_text("\n".join(lines) + "\n")

    # validation summary
    v = ["# Data Validation Summary",
         "",
         f"Final validated dataset: `data/tagged/<app>.csv.gz` (11 files), {n:,} reviews × {df.shape[1]} columns, "
         "every review posted on the Google Play Store for 11 apps from 1 April to 20 September 2026, collected by us "
         "(see `DATA_SOURCES.md`). Generated by `scripts/24_dataset_documentation.py`; numbers in "
         "`data/dataset_validation.json`.",
         "",
         "## From raw to final",
         "",
         "| Step | Reviews |",
         "|---|---:|"]
    v += [f"| {s} | {c:,} |" for s, c in steps]
    v += ["",
          "## Checks on the final dataset",
          "",
          f"All {len(checks)} checks passed." if out["all_checks_passed"] else "Some checks FAILED; see below.",
          "",
          "| Check | Result | Detail |",
          "|---|---|---|"]
    v += [f"| {c['check']} | {'Pass' if c['passed'] else 'FAIL'} | {c['detail']} |" for c in checks]
    v += ["",
          "## Known data issues (kept and flagged, not hidden)",
          "",
          "| Issue | Handling |",
          "|---|---|",
          "| Feed gap 21 Apr – 5 May for Swiggy, Blinkit, Domino's, Flipkart and Amazon: positive reviews drop 59–90%, "
          "negative reviews far less | Flagged (`feed_gap`); time series use counts and pass the gap to ARIMA as an input; "
          "share-based trends exclude the window |",
          "| Zomato has no reviews 23 Jul 22:00 – 25 Jul 12:30 (source outage, re-query returned the same) | "
          "24 Jul filled for forecasting only (`neg_reviews_filled`) |",
          f"| `app_version` empty for {av / n:.1%} of reviews | Kept; not a model input; release events use the rest |",
          "| Median review is 2 words | Kept; text mining uses reviews with ≥ 3 content words |",
          "",
          "## Suitability for each unit",
          "",
          "| Unit | Records and variables used |",
          "|---|---|"]
    v += [f"| {k} | {t} |" for k, t in units.items()]
    v += ["", "## Rows per app", "", "| App | Domain | Reviews |", "|---|---|---:|"]
    for app, c in df.app_name.value_counts().items():
        v.append(f"| {app} | {APP_DOMAIN[app]} | {c:,} |")
    (ROOT / "DATA_VALIDATION.md").write_text("\n".join(v) + "\n")

    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"] + " — " + c["detail"])


if __name__ == "__main__":
    main()
