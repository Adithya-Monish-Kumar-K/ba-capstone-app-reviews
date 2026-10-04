# Time-Series Analysis & Forecasting — Findings & Method

**Capstone Project — Stage 6: Time-Series (issue #111)**
*Owner: Person 5 · Input: `data/v1_most_relevant/app_reviews_tagged.csv` (14,988 reviews × 28 columns) · Code: `scripts/13_time_series_forecast.py`*

> **Data version note (Oct 2026).** This stage still uses the first dataset (moved to `data/v1_most_relevant/`). Review 1 now uses a new collection of 1,218,358 reviews of 11 apps; this analysis has not been rebuilt on it yet. The July 2026 "regime shift" handled below does not appear in the new data (see `../EDA.md` §6.3).

Outputs: 8 charts in `data/charts/timeseries/`, 5 tables in `data/timeseries/`, and every number quoted below in `data/timeseries_summary.json`.

---

## 1. Headline findings

1. **Only 21 weeks can be modelled.** The common window is 20 Apr – 13 Sep 2026: the complete weeks in which every app has ≥30 reviews. It holds 12,428 of the 14,988 reviews (82.9%). Outside it the weekly series is too thin to use ([chart 01](data/charts/timeseries/ts_01_weekly_volume.png)).
2. **The July 2026 sampling shift dominates every weekly series.** Pooled over the five apps, the share of 1–2★ reviews steps from 77.9% to 71.1%, issue-tagged reviews from 70.4% to 62.9%, negative sentiment from 59.9% to 54.3%, while median review length falls from 60 to 42 words and weekly volume rises 2.8× ([chart 02](data/charts/timeseries/ts_02_weekly_failure_trend.png)). `EDA.md` §6.1 shows most of this is a review-length effect of the collection, so it is not evidence that the apps improved.
3. **After the step there is no significant trend in any app's rating, low-star share, sentiment or issue rate** (Kendall τ, all p > 0.05 over the 11 post-shift weeks). The data supports forecasting a *level*, not a direction.
4. **Simple level models forecast the weekly 1–2★ share to within 1.2–7.1 pp** on a 6-week hold-out. Simple exponential smoothing (SES) beats the naive forecast in 19 of 24 series. The mean of all history is the worst model in 18 of 24 because it ignores the July step ([chart 05](data/charts/timeseries/ts_05_forecast_backtest_accuracy.png)).
5. **For Swiggy and Zomato the forecast error is already at the sampling-noise floor.** Their weekly 1–2★ share barely moves (post-shift weekly SD 1.2 and 1.7 pp). Myntra, Paytm and PhonePe are 3–5× more volatile (SD 5.6–6.8 pp) and their forecasts are correspondingly wider.
6. **The 4-week forecast is a flat continuation of the current level:** 1–2★ share of about 93% (Swiggy), 89% (Zomato), 58% (Myntra), 59% (Paytm), 38% (PhonePe) and 68% pooled ([chart 07](data/charts/timeseries/ts_07_forecast_low_star_by_app.png)). These describe the scraped `MOST_RELEVANT` sample, not the apps' user bases.
7. **No release-style failure spike is visible at weekly resolution.** One of 523 app-issue-weeks exceeds its baseline by more than 3 standard errors, against 0.7 expected by chance ([chart 04](data/charts/timeseries/ts_04_weekly_issue_categories.png)).

---

## 2. Data used

| Item | Value |
|---|---|
| Dataset | `data/app_reviews_tagged.csv` (Person 2's enriched master dataset). Read only; no re-tagging or re-scoring. |
| Date column | `review_date` (timestamp of the review, 2018-09-15 → 2026-09-18, no nulls, no stored timezone) |
| Why not `month` | `month` is derived from `review_date` and is too coarse: the common window has 6 months but 21 weeks. |
| Why not the model files | `data/engineered_features.npz` and `data/model_predictions.csv` carry no date or review ID, so they cannot be placed on a time axis. |
| Fields used | `app_name`, `score`, `sentiment_label`, `sentiment_compound`, `has_issue`, `issue_count`, the nine `issue_*` flags, `review_length` |

`data/eda/monthly_trend.csv` (Person 3) covers the same metrics by month. The weekly tables here are built from the review-level data with the same definitions so the two agree.

---

## 3. Weekly aggregation methodology

| Choice | Why |
|---|---|
| Weeks run **Monday–Sunday**, labelled by `week_start` | Fixed calendar weeks; each review belongs to exactly one week. |
| **Missing weeks are kept as rows** with `review_count = 0` and empty rates | A week with no sampled reviews is a real gap. Rates are never interpolated or forward-filled, since that would invent observations. |
| **Partial last week excluded** (14–20 Sep, 330 reviews) | The newest review is dated 18 Sep and daily counts collapse after 15 Sep. |
| **Common window** = latest run of complete weeks where every app has ≥30 reviews | Same n ≥ 30 floor as `EDA.md`. Result: 20 Apr – 13 Sep 2026, 21 weeks. |
| `post_regime_shift` flag from the week of 29 Jun | The week counts as post-shift when most of its days fall on or after 1 Jul 2026 (the break found in `EDA.md` §6.1). 10 weeks before, 11 after. |
| Rows sorted by date before aggregation | Chronological order is preserved in every table. |

Coverage outside the common window is too sparse for weekly analysis:

| App | First week | Weeks spanned | Weeks with no reviews | Weekly reviews inside window (min–max) |
|---|---|---:|---:|---:|
| Swiggy | 2025-08-11 | 58 | 28 | 39–237 |
| Zomato | 2026-03-16 | 27 | 0 | 50–231 |
| Myntra | 2024-11-04 | 98 | 70 | 47–312 |
| Paytm | 2018-09-10 | 419 | 93 | 38–251 |
| PhonePe | 2018-09-10 | 419 | 102 | 38–187 |

For Paytm and PhonePe, 394 and 397 of the 397 weeks outside the window have fewer than 30 reviews. Their 2018–2025 history is kept in the tables (flagged `in_common_window = False`) but is not charted or modelled.

---

## 4. Metrics generated

Per app (`weekly_app_metrics.csv`) and pooled over the five apps (`weekly_overall_metrics.csv`), for every week:

| Metric | Definition |
|---|---|
| `review_count` | Reviews dated in the week |
| `mean_rating`, `sd_rating` | Mean and standard deviation of `score` |
| `pct_low_star`, `n_low_star` | Share and count of reviews rated 1–2★. Same definition as Person 4's `is_problematic` target. |
| `pct_negative_sentiment`, `n_negative_sentiment` | Share and count with VADER `sentiment_label = Negative` |
| `pct_has_issue`, `n_has_issue` | Share and count with at least one issue tag |
| `mean_sentiment`, `mean_issue_count`, `median_length` | Mean VADER compound, mean number of tags, median words |
| `pct_<issue>`, `n_<issue>` (×9) | Share and count for each issue category (multi-label, so shares do not sum to 100) |
| Flags | `in_common_window`, `post_regime_shift`, `partial_week`, `n_ge_min` |

The pooled series weights each app by its number of sampled reviews that week.

---

## 5. Forecasting methodology

**What is forecast.** Four weekly metrics for each app and for the pooled series (24 series): `pct_low_star` (primary), `mean_rating`, `pct_negative_sentiment`, `pct_has_issue`.

**What is not forecast: review volume.** Each app was scraped to a fixed quota of 3,000 reviews from the `MOST_RELEVANT` feed (`DATA_SOURCES.md`). Weekly counts therefore show how the feed spread that quota over time, not how many reviews users posted. A volume forecast would predict the scraper, so volume is described (chart 01) and left out of the models.

**Models.** Five transparent methods, each producing a flat forecast from past weeks only:

| Model | Forecast |
|---|---|
| Naive | Last observed week |
| Mean of all history | Mean of every training week |
| Level-shift mean | Mean of the training weeks since the July sampling shift (a step model) |
| 4-week moving average | Mean of the last 4 training weeks |
| Simple exponential smoothing (SES) | Exponentially weighted level; starting level = mean of the first 4 weeks; smoothing weight α chosen on a 0.05–0.95 grid to minimise one-step squared error inside the training weeks |

**Why nothing more complex.** With 21 weekly points and a level shift in the middle there are no repeated yearly cycles to estimate seasonality, and too few points for ARIMA or regression models with several parameters. Trend models were not used because no post-shift trend is statistically supported (§8).

**Final forecasts** use SES for every series. The model was fixed before looking at hold-out scores rather than picked per series, to avoid choosing a winner on the test weeks. Forecasts cover the four weeks starting 14 Sep, 21 Sep, 28 Sep and 5 Oct 2026.

**Prediction intervals (80%).** Forecast ± 1.28 × the RMSE of that model's own past errors at the same horizon, measured by rolling through the window (minimum 8 training weeks; 13 errors at 1 week ahead, 10 at 4 weeks). Intervals are not allowed to narrow as the horizon grows, and are clipped to 0–100% or 1–5★.

---

## 6. Train / test setup

| Item | Value |
|---|---|
| Hold-out | Last 6 complete weeks: 3 Aug – 13 Sep 2026 |
| Scheme | Rolling origin. Each hold-out week is forecast 1, 2, 3 and 4 weeks ahead, each time from the weeks up to the origin only. |
| Shortest training history | 12 weeks (first origin: week of 6 Jul) |
| Forecasts per series and model | 24 (6 weeks × 4 horizons); 2,880 rows in `forecast_backtest.csv` |
| Metrics | MAE (1 week ahead, and averaged over 1–4 weeks), RMSE, bias, skill vs naive (1 − MAE / naive MAE) |
| Leakage controls | Chronological split, no shuffling; SES weight re-estimated on each training slice; no test week enters any training statistic |

One caveat: the 1 Jul break date used by the level-shift model comes from the EDA, which saw the full dataset. The break lies before every forecast origin, so it was observable at each origin, but its date was not re-estimated inside the backtest.

**Sampling-noise floor.** A weekly rate based on n reviews is itself uncertain. `forecast_accuracy.csv` reports the mean absolute error a perfect forecast of the true rate would still make from sample size alone (0.8 × the standard error, averaged over the hold-out weeks).

---

## 7. Important assumptions

* The sampling regime of July–September 2026 continues through the forecast weeks. A new change in what the `MOST_RELEVANT` feed returns would invalidate the forecasts.
* `review_date` is used as scraped; no timezone is stored, so week boundaries may be off by a few hours.
* Weeks in the common window are treated as comparable even though weekly sample size varies from 38 to 312 reviews per app.
* Forecast errors are treated as roughly normal when building intervals.
* Levels describe the scraped sample (69% 1★ overall against public ratings of 4.4–4.7★). Only relative movement is meaningful.

---

## 8. Results

### 8.1 The July step (weekly means, before → after)

| Series | % rated 1–2★ | Mean rating | % with issue tag | Median words |
|---|---|---|---|---|
| Swiggy | 97.8 → 93.4 | 1.11 → 1.23 | 89.4 → 81.9 | 74 → 53 |
| Zomato | 88.9 → 88.5 | 1.42 → 1.42 | 87.5 → 75.6 | 71 → 48 |
| Myntra | 63.6 → 54.3 | 2.38 → 2.77 | 78.8 → 62.6 | 67 → 46 |
| Paytm | 74.5 → 60.8 | 1.89 → 2.49 | 53.0 → 44.3 | 41 → 26 |
| PhonePe | 65.7 → 43.5 | 2.23 → 3.08 | 44.8 → 30.5 | 41 → 20 |
| All apps | 77.9 → 71.1 | 1.81 → 2.09 | 70.4 → 62.9 | 60 → 42 |

The raw rating changes match the monthly ones in `EDA.md` §6.1 (Swiggy +0.11, Zomato −0.03, Myntra +0.41, Paytm +0.64, PhonePe +0.80). The EDA's length adjustment leaves a positive change only for Paytm (+0.38★) and PhonePe (+0.18★).

### 8.2 Hold-out accuracy, % rated 1–2★ (MAE in pp, 1–4 weeks ahead)

| Series | Naive | All-history mean | Level-shift mean | 4-week MA | SES | Noise floor |
|---|---:|---:|---:|---:|---:|---:|
| Swiggy | 1.31 | 2.98 | **0.85** | 1.03 | 1.16 | 1.37 |
| Zomato | 2.06 | 1.25 | 1.21 | **1.18** | 1.23 | 1.71 |
| Myntra | 7.94 | 6.74 | **5.73** | 6.65 | 7.06 | 2.54 |
| Paytm | 5.04 | 13.13 | 6.13 | 5.35 | **4.77** | 3.54 |
| PhonePe | 5.04 | 17.72 | 5.97 | **4.89** | 5.34 | 3.31 |
| All apps | **2.65** | 5.85 | 2.68 | 2.75 | 3.05 | 1.18 |

Mean star rating (MAE in ★): SES scores 0.04 (Swiggy), 0.05 (Zomato), 0.26 (Myntra), 0.21 (Paytm), 0.20 (PhonePe), 0.10 (pooled). The other two metrics are in `forecast_accuracy.csv`.

Across all 24 series:

| Model | Mean rank (1 = best) | Times best | Mean skill vs naive |
|---|---:|---:|---:|
| Level-shift mean | 2.04 | 12 | +0.13 |
| 4-week moving average | 2.29 | 3 | +0.11 |
| SES | 2.67 | 5 | +0.08 |
| Naive | 3.58 | 3 | 0.00 |
| Mean of all history | 4.42 | 1 | −0.86 |

* The four models that track the recent level are close to each other. The all-history mean is far worse wherever the July step is large (Paytm, PhonePe).
* SES was not the best model: the level-shift mean ranks first. The differences are small next to the noise (a few tenths of a pp for most series), and SES needs no externally supplied break date, so it was kept as the declared model.
* SES over-forecasts the pooled failure rates (bias +2.3 pp for the 1–2★ share) because the pooled series kept drifting down during the hold-out weeks.
* The 80% intervals are conservative: 92.4% of 144 one-week-ahead hold-out actuals fall inside them. The error history behind the intervals includes the weeks around the July step.

### 8.3 Forecasts for the weeks of 14 Sep – 5 Oct 2026 (SES, 80% interval at week 4)

| Series | % rated 1–2★ | Interval | Mean rating | Interval |
|---|---:|---|---:|---|
| Swiggy | 93.2 | 89.0–97.4 | 1.25 | 1.12–1.37 |
| Zomato | 88.6 | 84.8–92.5 | 1.41 | 1.28–1.54 |
| Myntra | 58.1 | 45.3–70.9 | 2.74 | 2.22–3.25 |
| Paytm | 58.7 | 46.5–71.0 | 2.54 | 2.00–3.08 |
| PhonePe | 38.2 | 17.0–59.4 | 3.23 | 2.40–4.05 |
| All apps | 67.9 | 61.5–74.2 | 2.20 | 1.96–2.45 |

Charts: [06 pooled, four metrics](data/charts/timeseries/ts_06_forecast_overall.png), [07 1–2★ share by app](data/charts/timeseries/ts_07_forecast_low_star_by_app.png), [08 rating by app](data/charts/timeseries/ts_08_forecast_rating_by_app.png). All values are in `weekly_forecast.csv`.

Myntra's SES weight for the 1–2★ share is α = 0.05, so its forecast (58.1%) behaves like a long-run average and sits above its post-shift mean (54.3%). This is the main case where the level-shift mean (hold-out MAE 5.7 vs 7.1 pp) is the better description.

---

## 9. Important trends observed

* **One step, then flat.** Over the whole window most series show a strong monotonic trend (pooled 1–2★ share: Kendall τ = −0.52, p < 0.001). Within the 11 post-shift weeks no app has a significant trend in any of the four forecast metrics; the closest is PhonePe's 1–2★ share (τ = −0.46, p = 0.06). The "trend" is the step.
* **Review length keeps falling after July.** Pooled median length continues down after the shift (τ = −0.64, p = 0.009), and so does the pooled issue-tag rate (τ = −0.60, p = 0.010). Given the length effect documented in the EDA, the continuing fall in tag rate cannot be separated from the continuing fall in length.
* **Volume keeps rising after July** (pooled τ = +0.71, p = 0.002), from 734 to 1,069 reviews per week. This is consistent with the feed favouring recent reviews and is a further reason not to forecast volume.
* **Issue categories are stable within each regime.** After July only 2 of 45 app × issue series trend at p < 0.05 (Paytm Payment & Refund, τ = −0.49; PhonePe UI/UX & Update, τ = −0.67). About 2 would be expected by chance, so neither should be read as a finding on its own.
* **Spike scan.** Each app-issue-week was compared with that app's own rate in the same regime. One of 523 testable cells exceeds 3 standard errors: Swiggy Order Quality in the week of 29 Jun (35 of 188 reviews, 18.6% vs an 11.2% baseline, z = 3.3). That is the mixed week that straddles the sampling shift, and 0.7 such flags are expected by chance.

---

## 10. Per-app differences

| App | Level of 1–2★ share (post-shift) | Week-to-week SD (pp) | July step (pp) | What stands out |
|---|---:|---:|---:|---|
| Swiggy | 93.4% | 1.2 | −4.3 | Near the ceiling every week; forecast error at the noise floor. |
| Zomato | 88.5% | 1.7 | −0.4 | No step in rating or 1–2★ share, yet its issue-tag rate falls 11.9 pp: fewer tags without better ratings. |
| Myntra | 54.3% | 5.6 | −9.3 | Most volatile relative to its noise floor (SES error 2.8× the floor). |
| Paytm | 60.8% | 6.2 | −13.8 | Large step; Crash & Stability is the one major tag that does not fall (18.9% → 20.1%). |
| PhonePe | 43.5% | 6.8 | −22.2 | Largest step and widest forecast interval (17–59% by week 4). |

Issue fingerprints from the EDA hold week by week ([chart 04](data/charts/timeseries/ts_04_weekly_issue_categories.png)): Delivery Delay is 27% of post-shift weekly reviews at Swiggy and Zomato and about 1% at the UPI apps; Cancellation & Return is 37% at Myntra; Customer Support is the largest category for the delivery apps (46% Swiggy, 43% Zomato).

---

## 11. Limitations

* **21 weeks, one regime change.** No seasonality, holiday or release-cycle effect can be estimated. Forecasts are short-horizon level estimates.
* **The step is mostly a sampling artefact** and is not adjusted for here. The EDA's length standardisation cannot be repeated weekly: 38 of 420 app-week-length-band cells are empty and 65 hold fewer than 5 reviews.
* **Biased sample.** `MOST_RELEVANT` reviews over-represent complaints; rates are not population estimates.
* **Small weekly samples.** With 38–312 reviews per app-week, sampling noise alone gives 1–4 pp of error on the 1–2★ share. Swiggy and Zomato forecasts cannot be improved further with this data.
* **Six hold-out weeks.** Accuracy differences between the four level-tracking models are within noise; the ranking should not be over-read.
* **Many tests.** More than 100 trend tests and 523 spike tests were run without multiple-comparison correction; expected false positives are stated next to each count.
* **One snapshot** (scraped 20 Sep 2026). The forecasts cannot be checked against later weeks from this repository, and re-scraping would return a different relevance-ranked sample.
* **No version-level time series.** `app_version` is the reviewer's installed version, not a release date, so update impact cannot be dated from this data (`EDA.md` §6.2).

---

## 12. Combined insights across stages

**With Text Mining (`TEXT_MINING.md`).**
* The weekly VADER negative share moves with the weekly 1–2★ share: pooled correlation 0.82 in levels and 0.66 in week-to-week changes. This extends Person 2's review-level validation (Pearson r = 0.60) to the weekly aggregate. The link is strong for Myntra (0.90 on weekly changes) and PhonePe (0.71) and weak for Paytm (0.12) and Swiggy (0.32), where the 1–2★ share has little week-to-week variation to explain.
* The issue-tag rate is a weaker weekly signal (pooled 0.44 on weekly changes; Zomato 0.02). Zomato's tag rate fell 11.9 pp at the July shift while its ratings did not move, which fits the EDA finding that tag recall depends on review length.
* Customer Support stays the top category for Swiggy and Zomato in every regime, and Cancellation & Return for Myntra, matching Person 2's per-app "top reported issue".

**With EDA (`EDA.md`).**
* The weekly series reproduces the July 2026 regime shift and locates it more precisely: the change appears in the week of 29 Jun and is complete by the week of 6 Jul.
* The EDA advised against forecasting raw volume or tag rate across the step without modelling a level shift. The backtest confirms it: the model that ignores the step is worst in 18 of 24 series, and the explicit level-shift mean ranks best.
* The EDA found weak "bad release" evidence by app version and no crash or UI rise in the flagged versions. The weekly spike scan agrees: no category shows a release-style spike beyond chance.

**With Predictive Modeling (`MODEL_EVALUATION.md`).**
* `pct_low_star` is the weekly average of Person 4's `is_problematic` target. Its base rate is not constant: 74.1% overall, but 77.9% before and 71.1% after the July shift in the pooled weekly series, and from 43.5% (PhonePe) to 93.4% (Swiggy) by app after the shift.
* The classifiers were evaluated on a random stratified 80/20 split, which mixes weeks from both regimes in train and test. The reported scores (F1 0.93–0.94) therefore do not show how the models would perform on later weeks; a time-ordered split would be needed for that.
* If the classifiers were used to track the weekly problematic share, their error asymmetry would matter. On the test set the true share is 74.1%; Logistic Regression flags 70.6% (−3.5 pp) and Random Forest 80.0% (+5.9 pp). Those offsets are as large as the pooled July step (−6.8 pp) and larger than the pooled weekly forecast error of the level-tracking models (2.6–3.1 pp), so a classifier-based trend line would need calibration before being compared with the rating-based one.
* `review_length` is one of the model's structural features and is also the variable that changes at the July shift, so the feature distribution the models learned differs between the two regimes.

**Overall.** The four stages agree on the same picture: complaints in this sample are dominated by service and fulfilment issues that differ by app, they are stable from week to week, and the one large movement in the data (July 2026) comes from how reviews were sampled. Short-term monitoring of the weekly 1–2★ share is feasible with simple level models; detecting the impact of a specific app update is not supported by this dataset.

---

## 13. Reproducing the analysis

```bash
pip install pandas numpy scipy matplotlib
python scripts/13_time_series_forecast.py    # ~5 s; reads data/app_reviews_tagged.csv
```

The script has no random component: re-running it reproduces every table and the summary byte for byte. It can be run from any working directory.

| Output | Content |
|---|---|
| `data/timeseries/weekly_app_metrics.csv` | Per-app weekly metrics, all weeks, with flags (1,021 rows) |
| `data/timeseries/weekly_overall_metrics.csv` | Pooled weekly metrics (419 rows) |
| `data/timeseries/forecast_backtest.csv` | Every hold-out forecast: series, metric, model, horizon, origin, target, forecast, actual, error |
| `data/timeseries/forecast_accuracy.csv` | MAE, RMSE, bias, skill vs naive, noise floor and rank per series, metric and model |
| `data/timeseries/weekly_forecast.csv` | 4-week SES forecasts with 80% intervals |
| `data/timeseries_summary.json` | All quoted numbers, trend tests, spike scan and chart index |
| `data/charts/timeseries/ts_01` … `ts_08` | Volume, pooled failure trend, per-app trends, issue categories, backtest accuracy, pooled forecast, per-app forecasts (1–2★ share, rating) |
