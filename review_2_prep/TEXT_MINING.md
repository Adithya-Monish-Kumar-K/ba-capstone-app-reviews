# Text Mining Methodology & Empirical Findings

**Capstone Project — Stage 3: Text Mining (Issue Tagging + Sentiment Analysis)**  
*Owner: Person 2 (Resolving Issues #101, #102, #103)*

> **Data version note (Oct 2026).** The numbers below describe the first dataset (14,988 `MOST_RELEVANT` reviews of 5 apps, now in `data/v1_most_relevant/`). The tagging and sentiment scripts have been re-run on the new 1,218,358-review, 11-app collection, so `data/issue_tagging_summary.json`, `data/sentiment_summary.json` and the charts in `data/charts/` now reflect the new data. This document has not been updated yet.

---

## 1. Executive Summary & Pipeline Context

This stage converts unstructured user review text into structured analytical features for downstream analysis. Building directly upon Person 1's cleaned dataset (`data/app_reviews_clean.csv`, 14,988 rows), this pipeline:
1. **Tags multi-label issue categories** across 9 distinct operational dimensions using a domain-specific keyword taxonomy and rule application engine.
2. **Computes NLP sentiment intensity** via NLTK's VADER model, extracting granular positive, neutral, negative, and compound scores.
3. **Validates sentiment alignment** against ground-truth Play Store star ratings (1–5★), establishing strong statistical correlation ($r = 0.6016$).
4. **Delivers the unified dataset** (`data/app_reviews_tagged.csv`, 28 columns) that powers Person 3's Exploratory Data Analysis, Person 4's Predictive Modeling, and Person 5's App Update Impact Dashboard.

---

## 2. Issue Categorization Taxonomy

Consumer applications across Food Delivery (Swiggy, Zomato), E-Commerce (Myntra), and Fintech/UPI Payments (Paytm, PhonePe) present distinct operational failure modes. To capture these nuances, a 9-category taxonomy was engineered:

| Category | Indicator Column | Scope & Domain Keywords |
|---|---|---|
| **Crash & Stability** | `issue_crash_bugs_stability` | App crashes, freezes, lagging, black screens, force-close, loading failures, glitches, device environment / security checks (e.g. USB debugging false positives in fintech). |
| **Payment & Refund** | `issue_payment_refund` | Failed transactions, money deducted without order creation, UPI timeouts, refund delays/denials, double debits, wallet errors. |
| **Delivery & Delay** | `issue_delivery_delay` | Late deliveries, excessive wait times, delivery partner/rider delays, GPS live-tracking inaccuracies. |
| **Order Quality & Fulfillment** | `issue_order_quality_fulfillment` | Wrong items, missing items, damaged parcels, stale/cold food, fake deliveries (marked delivered without receiving). |
| **Cancellation & Return** | `issue_cancellation_return` | Inability to cancel, unfair cancellation fees, rejected return requests, reverse pickup delays, exchange failures. |
| **Customer Support** | `issue_customer_support` | Unresponsive customer care, unhelpful chatbot loops, closed tickets without resolution, rude executives, lack of human escalation. |
| **Account, Login & OTP** | `issue_account_login_otp` | OTP not received / delayed, login failures, blocked/locked accounts, KYC verification errors, SIM card binding errors. |
| **Pricing, Charges & Fraud** | `issue_pricing_charges_fraud` | Hidden platform/handling/surge fees, overcharging, coupons not applying, fake discounts, scam/looting allegations. |
| **UI/UX & Update Regressions** | `issue_ui_ux_update` | Post-update regressions ("worst update"), confusing interfaces, broken search/filters, intrusive full-screen ads, update prompt loops. |

### Rule Application Engine (`scripts/03_tag_issues.py`)
* **Multi-label Matching:** Reviews frequently report compounding issues (e.g., an app crash leading to a double payment and unhelpful support). The engine flags every category detected (`issue_<category> = 1`).
* **Primary Issue Attribution:** The engine counts pattern occurrences per category. The category with the highest pattern density is tagged as `primary_issue`.
* **Summary Fields:** Generates `issue_count` (integer count of triggered categories), `has_issue` (binary 0/1 flag), and `all_issues` (semicolon-delimited list).

---

## 3. VADER Sentiment Scoring Methodology

Sentiment intensity was quantified using NLTK's **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*) (`scripts/04_sentiment_analysis.py`).

### Why VADER?
1. **Tuned for Informal Consumer Text:** VADER's lexicon is specifically calibrated for social and review text, recognizing punctuation emphasis (e.g., "worst app ever!!!"), capitalization emphasis ("SCAM"), and degree modifiers ("extremely slow").
2. **Zero Training Leakage:** As a rule-based lexicon, it scores reviews independently without risk of overfitting or data leakage into Person 4's predictive models.

### Output Metrics
* `sentiment_neg`, `sentiment_neu`, `sentiment_pos`: Proportion of text falling into each emotional polarity (summing to 1.0).
* `sentiment_compound`: Normalized, weighted composite score ranging from **$-1.0$ (most negative)** to **$+1.0$ (most positive)**.
* `sentiment_label`: Standard research thresholds:
  * **Negative:** $\text{compound} \le -0.05$
  * **Neutral:** $-0.05 < \text{compound} < +0.05$
  * **Positive:** $\text{compound} \ge +0.05$

---

## 4. Empirical Findings & Validation

### 4.1 Statistical Validation vs Ground-Truth Star Ratings

Comparing VADER compound scores against user star ratings demonstrates robust alignment:

* **Pearson Correlation ($r$):** **`0.6016`** ($p < 10^{-10}$) — strong linear relationship.
* **Spearman Rank Correlation ($\rho$):** **`0.5647`** ($p < 10^{-10}$) — monotonic relationship.
* **3-Class Alignment Accuracy:** **`71.95%`** (1–2★ as Negative, 3★ as Neutral, 4–5★ as Positive).

| Play Store Rating | Review Count | Mean Compound | % Negative ($\le -0.05$) | % Positive ($\ge 0.05$) |
|:---:|:---:|:---:|:---:|:---:|
| **1★** | 10,366 | **`-0.3550`** | 71.9% | 24.1% |
| **2★** | 742 | **`-0.1355`** | 54.7% | 35.3% |
| **3★** | 579 | **`+0.0685`** | 41.3% | 49.7% |
| **4★** | 673 | **`+0.3990`** | 19.0% | 74.4% |
| **5★** | 2,628 | **`+0.6731`** | 6.7% | 90.4% |

> **Validation Note:** The 24.1% of 1★ reviews with positive VADER compound scores predominantly represent **sarcastic phrasing** (e.g., *"Great job Swiggy on charging me twice and never delivering"*), which lexicons treat as positive, or reviews where polite introductory remarks precede severe complaints.

---

### 4.2 Issue Frequency & Severity Hierarchy

Across the 14,988 reviews, **65.4%** triggered at least one issue tag. Measuring mean sentiment per category reveals which operational failures cause the deepest user dissatisfaction:

| Rank | Issue Category | Review Count | % of Dataset | Mean Sentiment ($\text{Compound}$) | % Negative Reviews |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **Order Quality & Fulfillment** | 996 | 6.6% | **`-0.5043`** | **81.8%** |
| 2 | **Pricing, Charges & Fraud** | 1,414 | 9.4% | **`-0.4445`** | **76.9%** |
| 3 | **Delivery Delay** | 2,301 | 15.3% | **`-0.4369`** | **76.7%** |
| 4 | **Payment & Refund** | 2,278 | 15.2% | **`-0.3794`** | **73.6%** |
| 5 | **Cancellation & Return** | 2,923 | 19.5% | **`-0.3483`** | **72.2%** |
| 6 | **Customer Support** | 5,150 | 34.4% | **`-0.3310`** | **70.6%** |
| 7 | **Account, Login & OTP** | 356 | 2.4% | **`-0.1699`** | **59.3%** |
| 8 | **Crash & Stability** | 938 | 6.3% | **`-0.1023`** | **53.1%** |
| 9 | **UI/UX & Update** | 364 | 2.4% | **`+0.1389`** | **39.6%** |

**Key Takeaways:**
* **Physical / Monetary Failures Provoke Peak Hostility:** Order quality failures (cold/spilled food, wrong items) and perceived financial fraud/hidden charges trigger the harshest sentiment (mean $< -0.44$).
* **Support is the Universal Pain Point:** 34.4% of all reviews cite unhelpful customer care, making it the highest-volume bottleneck across apps.
* **Technical Bugs Receive Less Hostile Language:** Crashes and login issues trigger factual, technical descriptions rather than profanity or hostile adjectives, resulting in milder negative polarity scores.

---

### 4.3 App Comparison Summary

| App | Total Reviews | Avg Star Rating | Mean Compound | % Negative | % Neutral | % Positive | Top Reported Issue |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Swiggy** | 3,000 | 1.21★ | **`-0.3795`** | 73.8% | 3.4% | 22.8% | Customer Support (1,438) |
| **Zomato** | 2,997 | 1.43★ | **`-0.2969`** | 68.5% | 3.6% | 27.9% | Customer Support (1,340) |
| **Paytm** | 2,998 | 2.06★ | **`-0.0352`** | 49.7% | 7.0% | 43.3% | Customer Support (807) |
| **Myntra** | 2,998 | 2.65★ | **`+0.0625`** | 47.3% | 1.8% | 50.9% | Cancellation & Return (1,203) |
| **PhonePe** | 2,995 | 2.46★ | **`+0.0813`** | 40.8% | 6.4% | 52.8% | Customer Support (648) |

*(Note: Reflects the `Sort.MOST_RELEVANT` collection bias disclosed in `DATA_SOURCES.md`).*

---

## 5. Visual Artifacts Generated

Publication-quality visualizations are saved in [`data/charts/`](file:///c:/Users/krish/Downloads/BA_Project/ba-capstone-app-reviews/data/charts/):

1. **`01_sentiment_validation_by_rating.png`**: Boxplot with diamond means demonstrating monotonic compound score progression from 1★ to 5★ ratings ($r = 0.602, \rho = 0.565$).
2. **`02_sentiment_distribution_by_app.png`**: Stacked bar comparison of Negative, Neutral, and Positive percentages per app.
3. **`03_sentiment_by_issue_category.png`**: Horizontal bar chart ranking the 9 issue categories by mean compound sentiment severity.
4. **`04_sentiment_trend_monthly.png`**: Monthly sentiment time-series tracking app trajectories across the last 12 months.

---

## 6. Handoff to Downstream Stages

The master dataset [`data/app_reviews_tagged.csv`](file:///c:/Users/krish/Downloads/BA_Project/ba-capstone-app-reviews/data/app_reviews_tagged.csv) (14,988 rows $\times$ 28 columns) is complete and verified.

### For Person 3 (EDA):
* Use the binary columns `issue_*` to plot issue frequency distributions by app and over time.
* Use `sentiment_compound` and `sentiment_label` for distribution plots and app sentiment benchmarking.
* Pre-computed aggregation tables are available in [`data/sentiment_aggregation_by_app.csv`](file:///c:/Users/krish/Downloads/BA_Project/ba-capstone-app-reviews/data/sentiment_aggregation_by_app.csv) and [`data/sentiment_aggregation_by_issue.csv`](file:///c:/Users/krish/Downloads/BA_Project/ba-capstone-app-reviews/data/sentiment_aggregation_by_issue.csv).

### For Person 4 (Predictive Modeling):
* The 9 binary issue columns (`issue_*`) and 4 continuous sentiment metrics (`sentiment_neg`, `sentiment_neu`, `sentiment_pos`, `sentiment_compound`) serve as pre-engineered structural features to predict `is_problematic` ($\text{score} \le 2$).

### For Person 5 (Time-Series & Dashboard):
* Use `month`, `app_name`, and `app_version` combined with `sentiment_compound` and `issue_count` to detect failure spikes following version releases.

---

## 7. Reproducing Stage 3

```bash
# 1. Install dependencies
pip install nltk pandas scipy matplotlib seaborn scikit-learn

# 2. Run Issue Tagging (generates issue tags)
python scripts/03_tag_issues.py

# 3. Run Sentiment Analysis (computes VADER scores, validation, & charts)
python scripts/04_sentiment_analysis.py
```
