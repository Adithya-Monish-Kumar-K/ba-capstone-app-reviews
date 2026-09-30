# Mobile App Update Impact & Failure Detection

**Capstone Project — Data collection & preprocessing owned by Person 1**

Analyzing Play Store reviews for five major Indian consumer apps (Swiggy, Zomato, Myntra, Paytm, PhonePe) to detect user-reported issues, track them over time, and predict which reviews signal real trouble.

## Team ownership

Each pipeline stage is owned end-to-end by one person — see `Team_Work_Split.pdf` for the full commit-by-commit plan.

| Stage | Owner | Status |
|---|---|---|
| Data Collection & Preprocessing | Person 1 | Done |
| Text Mining (Issue Tagging + Sentiment) | Person 2 | Done (issues #101–#103) |
| Exploratory Data Analysis (EDA) | Person 3 | Done (issues #104–#106 + follow-ups) |
| Predictive Modeling (Feature Engineering + Classifier) | Person 4 | Planned — issues #107–#110 |
| Time-Series + Dashboard + Final Assembly | Person 5 | Planned — issues #111–#114 |

## Stage contents

| Path | What it is | Owner |
|---|---|---|
| `scripts/01_scrape_reviews.py` | Play Store review collection (google-play-scraper, MOST_RELEVANT sort) | Person 1 |
| `scripts/02_clean_reviews.py` | Deduplication, empty/non-English filtering, month + review_length fields | Person 1 |
| `scripts/03_tag_issues.py` | Domain-specific issue categorization engine (9 categories, multi-label regex) | Person 2 |
| `scripts/04_sentiment_analysis.py` | VADER sentiment scoring, rating validation, aggregations, and chart generation | Person 2 |
| `data/app_metadata.csv` | Current real public rating/installs per app | Person 1 |
| `data/app_reviews_raw.csv` | 15,000 raw scraped reviews, 5 apps | Person 1 |
| `data/app_reviews_clean.csv` | Cleaned dataset (14,988 rows) | Person 1 |
| `data/cleaning_results.json` | Row counts through the cleaning pipeline | Person 1 |
| `data/app_reviews_tagged.csv` | Enriched master dataset (14,988 rows, 28 columns) with issue flags + VADER sentiment | Person 2 |
| `data/charts/` | Publication-ready visual charts (rating validation, app breakdown, issue severity, monthly trends) | Person 2 |
| `data/sentiment_summary.json` | Comprehensive statistical metrics for sentiment validation & aggregation | Person 2 |
| `TEXT_MINING.md` | Detailed text mining methodology, validation statistics, and empirical findings | Person 2 |
| `scripts/05_eda.py` | Exploratory analysis: 13 charts, statistical tests, coverage and July-2026 regime-shift audit | Person 3 |
| `data/charts/eda/` | 13 EDA charts (`eda_01_…` to `eda_13_…`) | Person 3 |
| `data/eda/monthly_trend.csv` | Monthly per-app metrics over all months with common-window / regime-shift flags: trend input for Person 5 | Person 3 |
| `data/eda/version_metrics.csv` | Rating per app version (≥30 reviews) with z-scores, a shortlist of unusual builds for Person 5 | Person 3 |
| `data/eda_summary.json` | Every number quoted in `EDA.md`, plus statistical tests | Person 3 |
| `EDA.md` | EDA methodology, findings, limitations, and implications for Persons 4–5 | Person 3 |

See `DATA_SOURCES.md` for data collection details, `TEXT_MINING.md` for text mining methodology and `EDA.md` for the exploratory analysis.

## Reproducing the pipeline

```bash
# Stage 1 & 2: Data Collection & Cleaning
pip install google-play-scraper pandas
python scripts/01_scrape_reviews.py   # pulls fresh data from Play Store
python scripts/02_clean_reviews.py    # dedupe, filter, derive fields

# Stage 3: Text Mining (Issue Tagging & Sentiment Analysis)
pip install nltk scipy matplotlib seaborn scikit-learn
python scripts/03_tag_issues.py       # applies 9-category issue taxonomy
python scripts/04_sentiment_analysis.py # computes VADER sentiment, validation, & charts

# Stage 4: Exploratory Data Analysis (reads app_reviews_tagged.csv, writes charts/tables/summary)
python scripts/05_eda.py              # ~10 s; 13 charts in data/charts/eda/, summary in data/eda_summary.json
```

## Handoff to Person 3 (EDA) & Person 4 (Predictive Modeling)

`data/app_reviews_tagged.csv` is the complete input dataset for downstream analysis:
* **Review metadata:** `app_id`, `app_name`, `review_id`, `score`, `content`, `thumbs_up`, `app_version`, `review_date`, `month`, `review_length`.
* **Issue indicators (9 binary columns):** `issue_crash_bugs_stability`, `issue_payment_refund`, `issue_delivery_delay`, `issue_order_quality_fulfillment`, `issue_cancellation_return`, `issue_customer_support`, `issue_account_login_otp`, `issue_pricing_charges_fraud`, `issue_ui_ux_update`.
* **Issue summaries:** `issue_count`, `has_issue`, `primary_issue`, `all_issues`.
* **VADER sentiment scores:** `sentiment_neg`, `sentiment_neu`, `sentiment_pos`, `sentiment_compound`, `sentiment_label`.

## EDA caveats every downstream stage should know (details in `EDA.md`)
* **Use the common window (Apr–Sep 2026) for cross-app comparisons.** Swiggy/Zomato/Myntra are ~100% 2026 reviews; only Paytm/PhonePe go back to 2018.
* **A sampling regime shift occurs in July 2026 in all five apps** (2.9× reviews/day, 27–50% shorter reviews). Apparent post-July drops in issue rates are mostly a review-length effect (chart 11, `data/eda_summary.json → regime_shift`).
* **`is_problematic` (score ≤ 2) has a 74.1% base rate**, so accuracy is a misleading metric for Person 4.
* **`issue_count` partly proxies review length** (Spearman 0.40) — include `review_length` and check the model is not only learning length.
* **Upvotes (`thumbs_up`) are extremely heavy-tailed** (Gini 0.97): use median or `log1p`.
