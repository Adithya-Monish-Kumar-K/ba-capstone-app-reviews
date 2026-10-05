# Data Sources & Collection Method

Required by the capstone rubric: *"The data source and data-collection method must be documented."*

## Source

**Google Play Store**, accessed via the [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) Python package. This calls Google's own internal Play Store review endpoint directly — not HTML scraping, no login required. No pre-built dataset (Kaggle/UCI/GitHub) was used anywhere in this project.

## Apps and package IDs

Eleven apps in three domains. All three domains share the same kinds of problems (payments, refunds, support, delivery or fulfilment, app crashes), so one issue taxonomy covers all of them.

| Domain | App | Package ID | Play category | Installs |
|---|---|---|---|---|
| Food & Grocery | Swiggy | `in.swiggy.android` | Food & Drink | 100M+ |
| Food & Grocery | Zomato | `com.application.zomato` | Food & Drink | 100M+ |
| Food & Grocery | Blinkit | `com.grofers.customerapp` | Food & Drink | 100M+ |
| Food & Grocery | Domino's | `com.Dominos` | Food & Drink | 100M+ |
| Shopping | Myntra | `com.myntra.android` | Shopping | 100M+ |
| Shopping | Flipkart | `com.flipkart.android` | Shopping | 1B+ |
| Shopping | Amazon | `in.amazon.mShop.android.shopping` | Shopping | 500M+ |
| Shopping | Meesho | `com.meesho.supply` | Shopping | 500M+ |
| Payments | Paytm | `net.one97.paytm` | Finance | 500M+ |
| Payments | PhonePe | `com.phonepe.app` | Finance | 1B+ |
| Payments | Google Pay | `com.google.android.apps.nbu.paisa.user` | Finance | 1B+ |

The app list lives in one place, `scripts/apps.py`, and every script reads it from there.

## Collection parameters

- **Window:** every review posted from **1 April 2026** to **20 September 2026** (both days included; `START_DATE` and `END_DATE` in `scripts/01_scrape_reviews.py`)
- **Sort method:** `Sort.NEWEST`, paginated backwards in time via `continuation_token` until a whole page is older than 1 April 2026
- **Volume:** 1,202,729 raw reviews; 1,137,987 after cleaning
- **Fields captured:** `review_id`, `score` (1–5★), `content` (review text), `thumbs_up`, `app_version`, `review_date`, plus `app_id`, `app_name`, `domain`
- **Language/region:** `lang="en"`, `country="in"`
- **Storage:** one gzipped CSV per app per stage (`data/raw/`, `data/clean/`, `data/tagged/`), so no single file exceeds GitHub's 100 MB limit

| App | Raw reviews | Reviews per day |
|---|---:|---:|
| Flipkart | 288,063 | 1,539 |
| Blinkit | 237,716 | 1,320 |
| Zomato | 159,399 | 870 |
| Meesho | 107,744 | 594 |
| PhonePe | 94,101 | 512 |
| Myntra | 87,160 | 493 |
| Swiggy | 78,562 | 432 |
| Paytm | 47,940 | 258 |
| Domino's | 46,298 | 252 |
| Amazon | 34,035 | 189 |
| Google Pay | 21,711 | 117 |
| **Total** | **1,202,729** | |

(Reviews per day are after cleaning.)

### Why a fixed date window with `NEWEST`

The scraper can also sort by `Sort.MOST_RELEVANT`. We tested that feed before collecting and found two problems:

1. **A hard ceiling.** In a test, `MOST_RELEVANT` ran out of reviews for Swiggy at 11,200 (the other apps were still returning reviews when the short test stopped at 8,200–8,800). `NEWEST` returned more than 100,000 reviews for several apps.
2. **Uneven, biased coverage.** It returns heavily-upvoted reviews from any year, so Paytm and PhonePe reached back to 2018 while the other apps were almost all 2026. It was also far more negative than the full review stream (69% 1★ in our test, against 17% with `NEWEST`).

`NEWEST` has no such ceiling. It returns *every* review in time order. Collecting everything since a fixed date gives every app **exactly the same time window**. Cross-app comparisons need no "common window", and busy apps simply contribute more reviews.

### How the scraper handles large pulls

- Each app is scraped independently and can run in parallel (`python scripts/01_scrape_reviews.py flipkart amazon`).
- A failed page is retried with backoff. An empty page mid-stream is retried before the scraper gives up, because the Play Store occasionally returns one (seen for Flipkart at 41,400 reviews).
- An app whose output file already exists is skipped, so an interrupted run can be restarted.

### Known data issues (disclosed, not hidden)

| Issue | What it is | How we handle it |
|---|---|---|
| **Short reviews dominate** | Median review is 2 words; 66% have 3 words or fewer ("good", "nice app") | Kept: they are real reviews and still carry a rating. Issue tags and text features naturally cover the longer reviews. |
| **Written reviews are harsher than the public rating** | Sample means sit 0.0–1.4★ below each app's Play Store rating, which also counts ratings without a written review | Compare apps and weeks with each other; never quote absolute satisfaction. |
| **21 Apr – 5 May 2026 feed gap** | For Swiggy, Blinkit, Domino's, Flipkart and Amazon, *positive* reviews drop by 59–90% while negative reviews drop only 3–37% (EDA chart 15). Negative reviews did not rise, so this is not a wave of complaints, and a scraper failure would remove all reviews alike, so the cause is on the Play Store side. | Rows kept and flagged; trend, version and July analyses exclude the window. |
| **One missing day for Zomato** | Zomato has no reviews from 23 Jul 22:00 to 25 Jul 12:30 (24 July is empty); re-querying the Play Store returned the same counts, so the gap is in the source, not the scraper | Kept as is; it lowers one week's Zomato volume and does not affect rates. Every other app has reviews on all 173 days. |
| **Missing app version** | `app_version` is empty for 13.4% of reviews | Kept; version is not a model feature. |

## Processing pipeline & ownership

Each stage is a separate person's responsibility, each reading the previous stage's output:

1. **Scrape** (Person 1) — `scripts/01_scrape_reviews.py` → `data/raw/<app>.csv.gz`, `data/app_metadata.csv`
2. **Clean** (Person 1) — dedupe; drop empty/very short reviews (< 3 characters), reviews with no letters at all (emoji or punctuation only) and non-English reviews (fewer than 85% of the letters are basic Latin a–z); derive `month`/`review_length` → `scripts/02_clean_reviews.py` → `data/clean/<app>.csv.gz` (1,137,987 rows). Language is judged on letters only, so an English review with emojis such as "good 👍" is kept. (An earlier version counted all characters, so emojis pushed about 76,000 English reviews below the 85% threshold; that was fixed before this analysis.)
3. **Tag + Sentiment** — rule-based issue-category keywords + VADER sentiment scoring → `data/tagged/<app>.csv.gz`
4. **EDA** (Persons 3 and 2) — `scripts/05_eda.py`, 15 charts, reading the tagged dataset
5. **Feature Engineering + Predictive Model** (Persons 4 and 5) — TF-IDF → Truncated SVD + structural features → Logistic Regression / Random Forest classifying `is_problematic` (score ≤ 2)
6. **Time-Series + Dashboard + Assembly** (Review 2) — weekly forecast models and interactive dashboard.

Shared helpers: `scripts/apps.py` (app list, domains, colours) and `scripts/data_io.py` (read/write the per-app files).
