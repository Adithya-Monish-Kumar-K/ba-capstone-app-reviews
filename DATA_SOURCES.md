# Data Sources & Collection Method

Required by the capstone rubric: *"The data source and data-collection method must be documented."*

## Source

**Google Play Store**, accessed via the [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) Python package. This calls Google's own internal Play Store review endpoint directly — not HTML scraping, no login required. No pre-built dataset (Kaggle/UCI/GitHub) was used anywhere in this project.

## Apps and package IDs

| App | Package ID |
|---|---|
| Swiggy | `in.swiggy.android` |
| Zomato | `com.application.zomato` |
| Myntra | `com.myntra.android` |
| Paytm | `net.one97.paytm` |
| PhonePe | `com.phonepe.app` |

## Collection parameters

- **Date collected:** 20 September 2026
- **Sort method:** `Sort.MOST_RELEVANT`, paginated via `continuation_token`
- **Volume:** 3,000 reviews per app (15,000 total)
- **Fields captured:** `review_id`, `score` (1–5★), `content` (review text), `thumbs_up`, `app_version`, `review_date`
- **Language/region:** `lang="en"`, `country="in"`

### Why `MOST_RELEVANT` instead of `NEWEST`

An initial pull using `Sort.NEWEST` (1,600 reviews/app) returned a date range of only ~4 days — these apps receive thousands of reviews daily, so recency-sorted pagination cannot reach back far enough for any time-series analysis. `Sort.MOST_RELEVANT` surfaces heavily-upvoted reviews regardless of age, reaching back to 2018 for some apps, which is what makes the monthly/weekly trend analysis in this project possible from a single scrape (no repeated/scheduled scraping needed).

### Known sampling bias (disclosed, not hidden)

Play Store users upvote complaint reviews far more than generic praise, so the `MOST_RELEVANT` feed is skewed negative relative to the true population:

| | |
|---|---|
| Real public rating of these 5 apps | 4.43 – 4.66 ★ |
| Scraped sample rating distribution | 69.2% are 1★, only 17.5% are 5★ |

This applies to **every downstream stage**, not just this one — it does not invalidate trend/correlation findings (which compare month-to-month or app-to-app within the sample), but absolute negative-review rates must never be quoted as representative of the apps' real user bases.

## Processing pipeline & ownership

Each stage is a separate person's responsibility, each reading the previous stage's output:

1. **Scrape** (Person 1) — `scripts/01_scrape_reviews.py` → `app_reviews_raw.csv`
2. **Clean** (Person 1) — dedupe, drop empty/non-English reviews, derive `month`/`review_length` → `scripts/02_clean_reviews.py` → `app_reviews_clean.csv` (14,988 rows, **not yet tagged**)
3. **Tag + Sentiment** (Person 2) — rule-based issue-category keywords + VADER sentiment scoring → `app_reviews_tagged.csv`
4. **EDA** (Person 3) — rating/volume/issue-frequency/trend charts, reading the tagged dataset
5. **Feature Engineering + Predictive Model** (Person 4) — TF-IDF → Truncated SVD + structural features → Logistic Regression / Random Forest classifying `is_problematic` (score ≤ 2)
6. **Time-Series + Dashboard + Assembly** (Person 5) — weekly forecast models, interactive dashboard, final notebook/report/deck

Only stage 1–2 is complete as of this branch. See `Team_Work_Split.pdf` for the full commit-by-commit plan for stages 3–6.
