# Mobile App Update Impact & Failure Detection

**Capstone Project — Review 1 (Units 1–2): Data Analysis and Predictive Modelling**

Analyzing Play Store reviews for five major Indian consumer apps (Swiggy, Zomato, Myntra, Paytm, PhonePe) to detect user-reported issues, track them over time, and predict which reviews signal real trouble.

## Contents

| Path | What it is |
|---|---|
| `docs/Review1_Report.pdf` | Full written report — problem statement, dataset & collection method, cleaning, EDA, feature engineering, model, evaluation, business findings |
| `docs/Review1_Presentation.pptx` | 13-slide team presentation deck |
| `docs/App_Update_Impact_Analysis.ipynb` | Executable Jupyter notebook — cleaning → EDA → feature engineering → model → evaluation, with all outputs pre-run |
| `data/app_reviews_raw.csv` | 15,000 raw scraped reviews, 5 apps |
| `data/app_reviews_tagged.csv` | Raw reviews + 10 rule-based issue-category flags |
| `data/app_reviews_clean.csv` | Final cleaned dataset used for modelling (14,988 rows) |
| `data/monthly_trend.csv` / `monthly_trend_reliable.csv` | Monthly avg rating + issue-mention rate per app (reliable = ≥30 reviews/month) |
| `data/app_metadata.csv` | Current real public rating/installs per app, for comparison against the sample |
| `data/analysis_results.json` | All numeric results (cleaning counts, correlations, model metrics, top terms) in one file |
| `figures/` | All 12 chart PNGs used in the report/notebook (rating distribution, review volume, issue frequency, monthly rating-vs-issue trend, confusion matrices, top terms, RF importance, ROC curve, PR curve, SVD projection, per-app issue breakdown, review length vs. rating) |

See `DATA_SOURCES.md` for the exact data collection method (required by the project rubric).

## Reproducing this

```bash
pip install google-play-scraper pandas scikit-learn matplotlib seaborn
python scripts/01_scrape_reviews.py   # pulls fresh data from Play Store
python scripts/02_tag_issues.py       # rule-based issue-category tagging
python scripts/03_analyze.py          # cleaning, EDA, feature engineering, model
```

## Key finding

Monthly issue-keyword mention rate is strongly negatively correlated with monthly average app rating for 4 of 5 apps (Myntra −0.80, Paytm −0.94, PhonePe −0.95, Swiggy −0.88) — a rising complaint rate is a leading indicator of a rating drop. A review-text classifier flags "problematic" (1–2★) reviews at 88.4% accuracy / 0.945 ROC-AUC.

**Important:** the scraped sample is complaint-enriched (69% 1★) relative to these apps' real public ratings (4.4–4.7★), due to the `MOST_RELEVANT` sort method used to get historical depth. See Section 2 of the report for full detail — this does not affect the correlation/trend findings, but absolute negative-review rates should never be quoted as representative of the real user base.
