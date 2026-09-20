# Mobile App Update Impact & Failure Detection

**Capstone Project — Data collection & preprocessing owned by Person 1**

Analyzing Play Store reviews for five major Indian consumer apps (Swiggy, Zomato, Myntra, Paytm, PhonePe) to detect user-reported issues, track them over time, and predict which reviews signal real trouble.

## Team ownership

Each pipeline stage is owned end-to-end by one person — see `Team_Work_Split.pdf` for the full commit-by-commit plan.

| Stage | Owner | Status |
|---|---|---|
| Data Collection & Preprocessing | Person 1 | Done (this branch) |
| Text Mining (Issue Tagging + Sentiment) | Person 2 | Planned — issues #62–#69 |
| Exploratory Data Analysis (EDA) | Person 3 | Planned — issues #70–#77 |
| Predictive Modeling (Feature Engineering + Classifier) | Person 4 | Planned — issues #78–#86 |
| Time-Series + Dashboard + Final Assembly | Person 5 | Planned — issues #87–#98 |

## This stage's contents

| Path | What it is |
|---|---|
| `scripts/01_scrape_reviews.py` | Play Store review collection (google-play-scraper, MOST_RELEVANT sort) |
| `scripts/02_clean_reviews.py` | Deduplication, empty/non-English filtering, month + review_length fields |
| `data/app_metadata.csv` | Current real public rating/installs per app |
| `data/app_reviews_raw.csv` | 15,000 raw scraped reviews, 5 apps |
| `data/app_reviews_clean.csv` | Cleaned dataset (14,988 rows) — **not yet tagged**; Person 2 adds issue tags + sentiment next |
| `data/cleaning_results.json` | Row counts through the cleaning pipeline |

See `DATA_SOURCES.md` for the exact collection method and a known sampling-bias caveat that applies to every later stage.

## Reproducing this stage

```bash
pip install google-play-scraper pandas
python scripts/01_scrape_reviews.py   # pulls fresh data from Play Store
python scripts/02_clean_reviews.py    # dedupe, filter, derive fields
```

## Handoff to Person 2

`data/app_reviews_clean.csv` is the input Person 2's tagging + sentiment scripts read from. It has `content`, `score`, `review_date`, `month`, `review_length`, `thumbs_up`, `app_version` — no issue flags or sentiment yet.
