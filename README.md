# Mobile App Update Impact & Failure Detection

**23CSE452 Business Analytics · Team 10**

This project analyzes Google Play Store reviews from five major Indian consumer apps:

* Swiggy
* Zomato
* Myntra
* Paytm
* PhonePe

The project combines **data collection, preprocessing, text mining, exploratory data analysis, predictive modelling, time-series forecasting, and an interactive dashboard**.

### Project Pipeline

**Reviews → Cleaning → Issue Tagging & Sentiment → EDA → Predictive Modelling → Time-Series Forecasting → Dashboard**

The goal is to identify major user-reported problems, detect reviews that are likely to be problematic, monitor complaint patterns over time, and provide business-oriented insights.

---

## Team

| # | Name                   | Roll Number      | Contribution                                       |
| - | ---------------------- | ---------------- | -------------------------------------------------- |
| 1 | Aditya Monish Kumar K  | CB.SC.U4CSE23103 | Data collection and preprocessing                  |
| 2 | Regella Krishna Saketh | CB.SC.U4CSE23649 | Text mining: issue tagging and sentiment           |
| 3 | Akshay KS              | CB.SC.U4CSE23104 | Exploratory data analysis                          |
| 4 | Harshini Vennela       | CB.SC.U4CSE23455 | Feature engineering and predictive modelling       |
| 5 | Kanishka D             | CB.SC.U4CSE23155 | Time-series analysis, dashboard and final assembly |

---

## Dataset

Reviews were collected from the **Google Play Store** using `google-play-scraper`.

| Property              | Details                                |
| --------------------- | -------------------------------------- |
| Apps                  | Swiggy, Zomato, Myntra, Paytm, PhonePe |
| Reviews collected     | 15,000                                 |
| Reviews per app       | 3,000                                  |
| Language              | English                                |
| Country               | India                                  |
| Sort order            | `MOST_RELEVANT`                        |
| Final cleaned reviews | 14,988                                 |
| Final tagged dataset  | 14,988 rows, 28 columns                |
| Collection date       | 20 September 2026                      |

The dataset was collected specifically for this project rather than downloaded from Kaggle or another public dataset repository.

### Important Sampling Note

The collected reviews are **not representative of all users of the five apps**.

The `MOST_RELEVANT` Play Store feed is complaint-heavy. In our sample:

* 69.2% of reviews are 1-star.
* 74.1% are classified as problematic using the 1–2 star definition.
* Public app ratings are much higher than the sample average.

Therefore, the results should mainly be used for **relative comparison between apps and weeks**, rather than estimating overall customer satisfaction.

---

## Key Results

### Text Mining

Nine issue categories were identified using a keyword-based multi-label approach:

1. Crash & Stability
2. Payment & Refund
3. Delivery Delay
4. Order Quality
5. Cancellation & Return
6. Customer Support
7. Account/Login/OTP
8. Pricing & Fraud
9. UI/UX & Update

**65.4%** of cleaned reviews received at least one issue tag.

**Customer Support** was the largest issue category at **34.4%** of reviews.

VADER sentiment had a **Pearson correlation of 0.60** with star rating.

### Exploratory Analysis

Different apps showed different complaint patterns:

* **Swiggy:** Customer Support and Delivery Delay
* **Zomato:** Customer Support and Delivery Delay
* **Myntra:** Cancellation & Return
* **Paytm:** Customer Support and Crash & Stability
* **PhonePe:** Customer Support

A major sampling change was observed in **July 2026**. Review volume increased substantially while review length decreased, causing issue rates to appear lower. This should not automatically be interpreted as a real improvement in app quality.

### Predictive Modelling

The classification target was:

> **Problematic = 1–2 star review**

Features included:

* TF-IDF text features
* Truncated SVD components
* Issue indicators
* Sentiment features
* Review length
* Upvotes

The final feature set contained **115 features**.

| Model               | Accuracy | Precision | Recall |    F1 |
| ------------------- | -------: | --------: | -----: | ----: |
| Logistic Regression |    89.3% |     94.9% |  90.4% | 92.6% |
| Random Forest       |    90.6% |     90.5% |  97.6% | 93.9% |

Random Forest achieved the higher F1 and recall, while Logistic Regression produced fewer false positives.

### Time-Series Forecasting

Weekly review metrics were created for the common period **20 April – 13 September 2026**.

Five simple forecasting approaches were compared:

* Naive
* Historical Mean
* Level-Shift Mean
* 4-Week Moving Average
* Simple Exponential Smoothing

There were **21 usable complete weeks**.

Simple Exponential Smoothing performed better than the naive forecast in **19 of 24 tested series**.

The final forecasts estimate the expected weekly share of **1–2 star reviews**. They should be interpreted as level forecasts rather than predictions of a strong future trend.

---

## Methods

| Stage               | Method                                                          | Main Output                     |
| ------------------- | --------------------------------------------------------------- | ------------------------------- |
| Data Collection     | `google-play-scraper`, English, India, `MOST_RELEVANT`          | Raw reviews                     |
| Cleaning            | Duplicate checking, empty/non-English filtering, derived fields | Clean dataset                   |
| Text Mining         | Keyword-based 9-category issue tagging                          | Issue indicators                |
| Sentiment           | VADER                                                           | Sentiment scores and labels     |
| EDA                 | Distributions, app comparisons, correlations, statistical tests | EDA charts and summaries        |
| Feature Engineering | TF-IDF → SVD + structural features                              | 115 model features              |
| Classification      | Logistic Regression and Random Forest                           | Problem-review predictions      |
| Time-Series         | Weekly aggregation and five baseline forecasting models         | Forecasts and backtesting       |
| Dashboard           | Streamlit + Plotly                                              | Interactive analytics dashboard |

---

## Repository Structure

```text
ba-capstone-app-reviews/
│
├── README.md
├── REPORT.md
├── DATA_SOURCES.md
├── TEXT_MINING.md
├── EDA.md
├── MODEL_EVALUATION.md
├── MODEL_EVALUATION_SUMMARY.md
├── TIME_SERIES.md
│
├── scripts/
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
│   └── 13_time_series_forecast.py
│
├── data/
│   ├── app_reviews_raw.csv
│   ├── app_reviews_clean.csv
│   ├── app_reviews_tagged.csv
│   ├── app_metadata.csv
│   ├── charts/
│   │   ├── eda/
│   │   └── timeseries/
│   ├── eda/
│   └── timeseries/
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
├── dashboard/
│   ├── app.py
│   ├── charts.py
│   ├── utils.py
│   └── requirements.txt
│
└── docs/
    ├── Final_Report.pdf
    └── Review_Presentation.key.pptx
```

---

## Dashboard

The project includes an interactive **Streamlit + Plotly dashboard**.

The dashboard provides:

* Overview
* Weekly Trends
* Forecast
* Sentiment & Issues
* Key Findings

It allows users to filter results by app, period and metric.

### Run the Dashboard

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r dashboard/requirements.txt

streamlit run dashboard/app.py
```

The dashboard will normally open at:

```text
http://localhost:8501
```

The dashboard uses the already-generated files in `data/`, so the complete analysis does not need to be rerun just to view the dashboard.

---

## Reproducing the Analysis

After activating the virtual environment and installing the required packages, the main analysis scripts can be run in sequence.

### Text Mining

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
```

### Time-Series Analysis

```bash
python scripts/13_time_series_forecast.py
```

### VADER Setup

If required:

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Collecting Fresh Reviews

Fresh data can be collected using:

```bash
python scripts/01_scrape_reviews.py
python scripts/02_clean_reviews.py
```

**Important:** Running the scraper again will produce a different review sample. Therefore, the numerical results in the report may not be reproduced exactly from a new scrape.

The scripts should be checked to ensure their output paths point to this repository's `data/` directory before collecting fresh data.

---

## Important Analytical Limitations

### Sampling Bias

The `MOST_RELEVANT` Play Store sample is complaint-heavy and should not be interpreted as the actual satisfaction distribution of all app users.

### July 2026 Sampling Change

A sampling regime shift occurred in July 2026 across all five apps. Review volume increased while review length decreased. Apparent improvements in issue rates therefore need to be interpreted carefully.

### Classification Target

The classifier predicts whether a review is **1–2 stars**, which is used as a proxy for a problematic review.

It does **not** directly confirm that an actual software failure occurred.

### Text Mining Limitations

The issue tagger uses keyword patterns and can miss context, sarcasm, or topics that do not contain the expected keywords.

VADER sentiment can also misclassify sarcastic or politely worded negative reviews.

### Time-Series Limitations

Only 21 complete weeks were available for the common modelling period. The forecasts therefore provide short-term level estimates rather than strong long-term trend predictions.

### Upvote Distribution

`thumbs_up` is extremely heavy-tailed. It should therefore be interpreted carefully and transformed appropriately when used for modelling.

---

## Project Documents

| Document                                                            | Description                                  |
| ------------------------------------------------------------------- | -------------------------------------------- |
| [`REPORT.md`](REPORT.md)                                            | Full project report                          |
| [`Final_Report.pdf`](docs/Final_Report.pdf)                         | Final project report in PDF format           |
| [`Review_Presentation.key.pptx`](docs/Review_Presentation.key.pptx) | Project presentation                         |
| [`DATA_SOURCES.md`](DATA_SOURCES.md)                                | Data collection and sampling details         |
| [`TEXT_MINING.md`](TEXT_MINING.md)                                  | Issue tagging and sentiment analysis         |
| [`EDA.md`](EDA.md)                                                  | Exploratory data analysis                    |
| [`MODEL_EVALUATION.md`](MODEL_EVALUATION.md)                        | Predictive modelling methodology and results |
| [`MODEL_EVALUATION_SUMMARY.md`](MODEL_EVALUATION_SUMMARY.md)        | Short model evaluation summary               |
| [`TIME_SERIES.md`](TIME_SERIES.md)                                  | Weekly analysis and forecasting              |

---

## Conclusion

This project combines review mining, sentiment analysis, predictive modelling and time-series forecasting into a single business analytics workflow.

The analysis shows that:

* Customer support is a major complaint category across the dataset.
* Different apps have different dominant problem areas.
* Problematic reviews can be classified with an F1 score of approximately **0.93–0.94**.
* Weekly low-rating share can be forecast reasonably well using simple methods.
* Sampling changes can create misleading improvements if data collection is not monitored.
* An interactive dashboard makes the results easier to monitor and interpret.

The project therefore demonstrates how unstructured app reviews can be converted into **actionable business insights for monitoring, prioritization and short-term forecasting**.
