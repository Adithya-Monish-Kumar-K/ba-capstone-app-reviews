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

An initial pull using `Sort.NEWEST` (1,600 reviews/app) returned a date range of only ~4 days — these apps receive thousands of reviews daily, so recency-sorted pagination cannot reach back far enough for any time-series analysis. `Sort.MOST_RELEVANT` surfaces heavily-upvoted reviews regardless of age, reaching back to 2018 for some apps, which is what makes the monthly trend analysis in this project possible from a single scrape (no repeated/scheduled scraping was needed).

### Known sampling bias (disclosed, not hidden)

Play Store users upvote complaint reviews far more than generic praise, so the `MOST_RELEVANT` feed is skewed negative relative to the true population:

| | |
|---|---|
| Real public rating of these 5 apps | 4.43 – 4.66 ★ |
| Scraped sample rating distribution | 69.2% are 1★, only 17.5% are 5★ |

This is disclosed explicitly in the report (Section 2) and notebook (Section 1). It does not invalidate the trend/correlation findings (which compare month-to-month within the sample) but means absolute negative-review rates must never be quoted as representative of the apps' real user bases.

## Processing pipeline

1. **Scrape** — `scripts/01_scrape_reviews.py` → `app_reviews_raw.csv`
2. **Tag** — rule-based regex keyword matching for 10 issue categories → `app_reviews_tagged.csv`
3. **Clean** — dedupe, drop empty/non-English reviews → `app_reviews_clean.csv` (14,988 rows)
4. **Aggregate** — monthly avg rating + issue-mention rate per app (≥30 reviews/month filter for reliability) → `monthly_trend_reliable.csv`
5. **Model** — TF-IDF → Truncated SVD (50 components) + structural features → Logistic Regression / Random Forest classifying `is_problematic` (score ≤ 2)

Full detail in `docs/Review1_Report.pdf` and `docs/App_Update_Impact_Analysis.ipynb`.
