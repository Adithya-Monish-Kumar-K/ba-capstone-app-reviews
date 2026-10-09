# Mobile App Update Impact & Failure Detection

**23CSE452 Business Analytics · Team 10**

This project analyzes Google Play Store reviews from 11 major Indian consumer apps in three domains:

* **Food & Grocery:** Swiggy, Zomato, Blinkit, Domino's
* **Shopping:** Myntra, Flipkart, Amazon, Meesho
* **Payments:** Paytm, PhonePe, Google Pay

This repository is set up for **Review 1: data collection, preprocessing, exploratory data analysis and predictive modelling**. Review 2 adds text mining, time-series forecasting and an interactive dashboard.

### Project Pipeline

**Reviews → Cleaning → Issue Tagging & Sentiment → EDA → Predictive Modelling** (Review 1) → Time-Series Forecasting → Dashboard (Review 2)

The goal is to identify major user-reported problems, detect reviews that are likely to be problematic, measure how app versions affect ratings and complaints, monitor complaint patterns over time, and provide business-oriented insights.

* **Problem review** = a review rated 1–2 stars; the 9 issue tags name the failure when the text mentions one.
* **Update impact** = how ratings and complaints change between app versions: version-level comparison in Review 1 (EDA chart 11), before/after each release in Review 2.

---

## Team

| # | Name                   | Roll Number      | Contribution                                       |
| - | ---------------------- | ---------------- | -------------------------------------------------- |
| 1 | Aditya Monish Kumar K  | CB.SC.U4CSE23103 | Data collection and preprocessing                  |
| 2 | Regella Krishna Saketh | CB.SC.U4CSE23649 | Exploratory data analysis (with Akshay)            |
| 3 | Akshay KS              | CB.SC.U4CSE23104 | Exploratory data analysis (with Saketh)            |
| 4 | Harshini Vennela       | CB.SC.U4CSE23455 | Feature engineering and predictive modelling (with Kanishka) |
| 5 | Kanishka D             | CB.SC.U4CSE23155 | Feature engineering and predictive modelling (with Harshini) |

---

## Dataset

Reviews were collected from the **Google Play Store** using `google-play-scraper`: every English-language review from India posted since **1 April 2026**.

| Property              | Details                                                        |
| --------------------- | -------------------------------------------------------------- |
| Apps                  | 11 (4 Food & Grocery, 4 Shopping, 3 Payments)                  |
| Window                | 1 April – 20 September 2026, identical for every app           |
| Sort order            | `NEWEST`, paginated back to the start date                     |
| Language / Country    | English / India                                                |
| Raw reviews           | 1,202,729                                                      |
| Final cleaned reviews | 1,137,987                                                      |
| Final tagged dataset  | 1,137,987 rows, 29 columns                                     |
| Largest / smallest app | Flipkart 266,285 / Google Pay 20,320 (after cleaning)         |
| Storage               | one gzipped CSV per app per stage: `data/raw/`, `data/clean/`, `data/tagged/` |

The dataset was collected specifically for this project rather than downloaded from Kaggle or another public dataset repository. See `DATA_SOURCES.md` for the full method.

### Important Data Notes

* **Most reviews are short and positive.** 67% are 5-star, 17% are 1-star, and the median review is 2 words. Only 19.5% are problematic (1–2 stars).
* **Written reviews are harsher than the public rating**, which also counts ratings without text (gap of 0.0–1.4 stars by app). Use the results for **relative comparison between apps and weeks**, not as overall customer satisfaction.
* **21 April – 5 May 2026 feed gap:** for Swiggy, Blinkit, Domino's, Flipkart and Amazon, positive reviews drop 59–90% while negative reviews drop only 3–37%. The rows are kept and flagged; trend analyses exclude the window.

---

## Key Results

### Issue Tagging and Sentiment (inputs to EDA and the model)

Each review is tagged with nine keyword-based issue categories (multi-label) and scored with VADER sentiment:

1. Crash & Stability
2. Payment & Refund
3. Delivery Delay
4. Order Quality
5. Cancellation & Return
6. Customer Support
7. Account/Login/OTP
8. Pricing & Fraud
9. UI/UX & Update

### Exploratory Analysis

* **Amazon and Swiggy stand out:** 54% and 35% of their reviews are 1–2 stars, against 10–24% for the other apps. PhonePe (9.8%) and Myntra (10.3%) are the mildest.
* **Each domain has its own problem profile:** Customer Support leads Food & Grocery; Cancellation & Return leads Shopping; Payments has Crash & Stability and Customer Support, tied. Delivery Delay is about 50× more common in Food & Grocery than in Payments.
* **Apps in the same domain differ widely** (EDA chart 14): Swiggy 35% vs Zomato 19% in Food & Grocery, Amazon 54% vs Myntra 10% in Shopping, Google Pay 23% vs PhonePe 10% in Payments. Support and delivery lead the food apps, support and returns the shopping apps, and crashes the two weaker payment apps.
* **Most complaints that name an issue are about the service, not the app:** 91% of tagged 1–2★ reviews name a service problem and 7% an app problem (EDA chart 15). The payment apps are the exception (10–13% app-only at Google Pay and Paytm).
* **Customer Support** is the most common issue (4.1% of reviews) and the most damaging (1.23★ average).
* **Late-April feed gap:** a "% negative" metric jumps for five apps because positive reviews are missing, not because complaints rose. Counts of negative reviews per day are far less affected.
* **No mid-year shift:** review volume and length stay stable from April–June to July–September, so a time-based test split is safe.

15 charts in `data/charts/eda/`; details in `EDA.md`.

### Predictive Modelling

The classification target was:

> **Problematic = 1–2 star review** (19.5% of reviews)

Features included:

* TF-IDF text features (20,000 terms; no stop-word list, so negations such as "not good" are kept)
* Truncated SVD components (200)
* Issue indicators
* Sentiment features
* Review length
* Upvotes

The final feature set contained **215 features** for **1,137,987 reviews**.

| Model                   | Accuracy | Precision | Recall |    F1 | PR-AUC |
| ----------------------- | -------: | --------: | -----: | ----: | -----: |
| Majority-class baseline |    80.5% |      0.0% |   0.0% |  0.0% |  0.195 |
| Logistic Regression     |    92.2% |     76.7% |  85.8% | 81.0% |  0.871 |
| Random Forest           |    91.4% |     73.7% |  87.0% | 79.8% |  0.873 |

The two models are close: Logistic Regression is more accurate and precise at the default threshold (16% fewer false alarms), while Random Forest has the higher recall and PR-AUC. Trained on April–August and tested on 1–20 September, PR-AUC drops by less than 0.01 (Random Forest 0.866). Payment apps are the hardest domain (PR-AUC 0.75). 95% bootstrap intervals are about ±0.003 PR-AUC. The text components alone reach PR-AUC 0.85, more than the sentiment scores alone (0.74–0.80), and the settings we use are the best or within 0.005 PR-AUC of the best on a validation split. Details in `MODEL_EVALUATION.md`.

---

## Methods

| Stage               | Method                                                          | Main Output                     |
| ------------------- | --------------------------------------------------------------- | ------------------------------- |
| Data Collection     | `google-play-scraper`, English, India, `NEWEST` since 1 Apr 2026 | Raw reviews                     |
| Cleaning            | Duplicate, empty, no-letter and non-English filtering; derived fields | Clean dataset              |
| Text Mining         | Keyword-based 9-category issue tagging                          | Issue indicators                |
| Sentiment           | VADER                                                           | Sentiment scores and labels     |
| EDA                 | Distributions, app comparisons, correlations, statistical tests | EDA charts and summaries        |
| Feature Engineering | TF-IDF → SVD + structural features                              | 215 model features              |
| Classification      | Logistic Regression and Random Forest                           | Problem-review predictions      |

---

## Repository Structure

```text
ba-capstone-app-reviews/
│
├── README.md
├── REPORT.md
├── DATA_SOURCES.md
├── EDA.md
├── MODEL_EVALUATION.md
├── MODEL_EVALUATION_SUMMARY.md
├── requirements.txt
│
├── scripts/
│   ├── apps.py                  # the 11 apps, domains and colours (single source of truth)
│   ├── data_io.py               # read/write the per-app data files
│   ├── 01_scrape_reviews.py
│   ├── 02_clean_reviews.py
│   ├── 03_tag_issues.py
│   ├── 04_sentiment_analysis.py
│   ├── 05_eda.py
│   ├── 05_feature_engineering.py
│   ├── 06_model_training.py
│   ├── 07_model_evaluation_charts.py
│   ├── 08_model_interpretation.py
│   ├── 09_model_performance_comparison.py
│   ├── 10_auc_summary.py
│   ├── 11_prediction_error_summary.py
│   ├── 12_normalized_confusion_matrices.py
│   ├── 13_model_robustness_checks.py
│   ├── 14_predict_review.py     # classify new review text with the saved models
│   ├── 15_time_series_data.py   # Review 2: daily/weekly series, release events, STL, ADF, ACF/PACF
│   ├── 16_text_preprocessing.py # Review 2: cleaned complaint/praise corpora, document-term matrices
│   └── 19_forecasting.py        # Review 2: ARIMA/SARIMAX forecasts with rolling backtest, baselines
│
├── data/
│   ├── raw/<app>.csv.gz         # scraped reviews, one file per app
│   ├── clean/<app>.csv.gz       # cleaned
│   ├── tagged/<app>.csv.gz      # + issue tags and VADER sentiment (input to EDA and modelling)
│   ├── app_metadata.csv
│   ├── charts/                  # sentiment charts (01–04), eda/, timeseries/ and textmining/
│   ├── eda/                     # monthly, weekly and version tables
│   ├── timeseries/              # Review 2: daily/weekly series, release events, patterns
│   └── textmining/              # Review 2: cleaned corpus and document-term matrices
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── structural_scaler.pkl
│   ├── svd_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── figures/
│   └── model evaluation charts
│
├── dashboard/                   # Review 2 Streamlit dashboard (see dashboard/README.md)
│   ├── app.py
│   ├── common.py
│   └── pages/
│
├── notebooks/
│   └── Review_1_Analysis.ipynb
│
├── docs/
│   ├── Review_1_Presentation.pptx
│   └── Review_1_Contribution_Summary.md
│
└── review_2_prep/               # Review 2 workspace
```

---

## Reproducing the Analysis

After activating the virtual environment and installing the required packages, the main analysis scripts can be run in sequence.

### Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

`data/engineered_features.npz`, `data/target.npy` and `data/model_row_index.csv.gz` are not stored in git (about 570 MB). Scripts 06–13 read them, so run `scripts/05_feature_engineering.py` first.

### Issue Tagging and Sentiment

```bash
python scripts/03_tag_issues.py
python scripts/04_sentiment_analysis.py
```

### Exploratory Data Analysis

```bash
python scripts/05_eda.py
```

### Feature Engineering and Classification

```bash
python scripts/05_feature_engineering.py
python scripts/06_model_training.py
python scripts/07_model_evaluation_charts.py
python scripts/08_model_interpretation.py
python scripts/09_model_performance_comparison.py
python scripts/10_auc_summary.py
python scripts/11_prediction_error_summary.py
python scripts/12_normalized_confusion_matrices.py
python scripts/13_model_robustness_checks.py   # confidence intervals, ablation, settings check (~20 min)
```

### Classify a New Review

```bash
python scripts/14_predict_review.py "Refund not received, customer care never replies"
```

It prints the issue tags, the sentiment scores and each model's probability that the review is problematic (rated 1–2★).

### Review 2: Time Series, Text Mining and Dashboard

```bash
python scripts/15_time_series_data.py      # time-series data preparation (~10 s)
python scripts/16_text_preprocessing.py    # text-mining data preparation (~3 min; downloads NLTK WordNet once)
python scripts/19_forecasting.py           # ARIMA forecasts and rolling backtest (~1 min; needs scripts/15 first)
streamlit run dashboard/app.py             # interactive dashboard
```

Method details: `TIME_SERIES.md` and `TEXT_MINING.md`. Dashboard structure and how to add a page: `dashboard/README.md`.

### VADER Setup

If required:

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Collecting Fresh Reviews

Fresh data can be collected using:

```bash
python scripts/01_scrape_reviews.py                    # all 11 apps, ~1 hour
python scripts/01_scrape_reviews.py flipkart amazon    # a subset; run several in parallel to finish faster
python scripts/02_clean_reviews.py
```

The scraper collects every review from `START_DATE` (1 April 2026) to `END_DATE` (20 September 2026) and skips apps whose file already exists in `data/raw/`; delete a file to re-scrape that app. Apps are listed in `scripts/apps.py`.

**Important:** Running the scraper again will produce a different review set (new reviews, edited or deleted ones). Therefore, the numerical results in the report may not be reproduced exactly from a new scrape.

---

## Important Analytical Limitations

### Written Reviews vs Public Rating

Written reviews sit 0.0–1.4 stars below each app's public rating, which also counts ratings without text. The results should not be interpreted as the satisfaction of all app users.

### Late-April Feed Gap

From 21 April to 5 May 2026, positive reviews drop 59–90% for five apps while negative reviews drop only 3–37%. Share-based metrics (% negative, mean rating) are distorted in this window; counts of negative reviews per day are far less affected.

### Short Reviews

The median review is 2 words. Issue tags and text features carry little information for these reviews, so only 10% of reviews have an issue tag.

### Classification Target

The classifier predicts whether a review is **1–2 stars**, which is used as a proxy for a problematic review.

It does **not** directly confirm that an actual software failure occurred.

### Issue Tagging and Sentiment Limitations

The issue tagger uses keyword patterns and can miss context, sarcasm, or topics that do not contain the expected keywords.

VADER sentiment can also misclassify sarcastic or politely worded negative reviews.

### Upvote Distribution

`thumbs_up` is extremely heavy-tailed. It should therefore be interpreted carefully and transformed appropriately when used for modelling. It is also unknown for a brand-new review (0), so a live triage system should not rely on it; its weight in the models is small.

### Language Filter and Hinglish

The cleaning step keeps a review when at least 85% of its letters are Latin (a–z). Hindi written in Roman letters (Hinglish) therefore stays in the data, and VADER, an English lexicon, scores it near neutral.

### Issue Tag Details

`primary_issue` is the category with the most keyword hits; a tie goes to the first category in the taxonomy (Crash & Stability). It is only a convenience column: all analysis uses the nine 0/1 issue flags.

---

## Project Documents

| Document                                                            | Description                                  |
| ------------------------------------------------------------------- | -------------------------------------------- |
| [`REPORT.md`](REPORT.md)                                            | Review 1 project report                      |
| [`Review_1_Analysis.ipynb`](notebooks/Review_1_Analysis.ipynb)      | Review 1 notebook (executed, end to end)     |
| [`Review_1_Presentation.pptx`](docs/Review_1_Presentation.pptx)     | Review 1 presentation                        |
| [`Review_1_Contribution_Summary.md`](docs/Review_1_Contribution_Summary.md) | Review 1 individual contribution summary |
| [`DATA_SOURCES.md`](DATA_SOURCES.md)                                | Data collection and sampling details         |
| [`EDA.md`](EDA.md)                                                  | Exploratory data analysis                    |
| [`MODEL_EVALUATION.md`](MODEL_EVALUATION.md)                        | Predictive modelling methodology and results |
| [`MODEL_EVALUATION_SUMMARY.md`](MODEL_EVALUATION_SUMMARY.md)        | Short model evaluation summary               |
| [`TIME_SERIES.md`](TIME_SERIES.md)                                  | Review 2 Method 2: time-series analysis      |
| [`TEXT_MINING.md`](TEXT_MINING.md)                                  | Review 2 Method 1: text mining               |
| [`dashboard/README.md`](dashboard/README.md)                        | Review 2 interactive dashboard               |
| [`review_2_prep/`](review_2_prep/)                                  | Review 2 workspace                           |

---

## Conclusion

Review 1 combines our own data collection, cleaning, issue tagging, sentiment scoring, exploratory analysis and predictive modelling into one business analytics workflow.

The analysis shows that:

* Customer support is a major complaint category across the dataset.
* Different apps have different dominant problem areas.
* Problematic reviews (19.5% of all reviews) can be found with over 91% accuracy: 87% recall at 74% precision with Random Forest, or 77% precision at 86% recall with Logistic Regression (PR-AUC **0.87**), and the models hold up on later weeks.
* Collection effects (the late-April feed gap) can look like real changes if data collection is not checked.

It shows how unstructured app reviews can be turned into **actionable business insights for monitoring and prioritization**; Review 2 adds forecasting and an interactive dashboard.
