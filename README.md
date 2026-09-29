# Mobile App Update Impact & Failure Detection

**Capstone Project — Data collection & preprocessing owned by Person 1**

Analyzing Play Store reviews for five major Indian consumer apps (Swiggy, Zomato, Myntra, Paytm, PhonePe) to detect user-reported issues, track them over time, and predict which reviews signal real trouble.

## Team ownership

Each pipeline stage is owned end-to-end by one person — see `Team_Work_Split.pdf` for the full commit-by-commit plan.

| Stage | Owner | Status |
|---|---|---|
| Data Collection & Preprocessing | Person 1 | Done |
| Text Mining (Issue Tagging + Sentiment) | Person 2 | Done (issues #101–#103) |
| Exploratory Data Analysis (EDA) | Person 3 | Planned — issues #104–#106 |
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

See `DATA_SOURCES.md` for data collection details and `TEXT_MINING.md` for complete text mining methodology.

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
```

## Handoff to Person 3 (EDA) & Person 4 (Predictive Modeling)

`data/app_reviews_tagged.csv` is the complete input dataset for downstream analysis:
* **Review metadata:** `app_id`, `app_name`, `review_id`, `score`, `content`, `thumbs_up`, `app_version`, `review_date`, `month`, `review_length`.
* **Issue indicators (9 binary columns):** `issue_crash_bugs_stability`, `issue_payment_refund`, `issue_delivery_delay`, `issue_order_quality_fulfillment`, `issue_cancellation_return`, `issue_customer_support`, `issue_account_login_otp`, `issue_pricing_charges_fraud`, `issue_ui_ux_update`.
* **Issue summaries:** `issue_count`, `has_issue`, `primary_issue`, `all_issues`.
* **VADER sentiment scores:** `sentiment_neg`, `sentiment_neu`, `sentiment_pos`, `sentiment_compound`, `sentiment_label`.
