# Time-Series Analysis — Review 2, Method 2

**Capstone Project — Review 2 (Unit 3): Time-Series Analysis**
*Input: `data/tagged/*.csv.gz` (1,137,987 reviews, 11 apps, 1 Apr – 20 Sep 2026)*

The question: how many 1–2★ reviews will each app get in the coming weeks, and did a new app version make things worse? This document grows stage by stage: data preparation (#131), ARIMA forecasting (#135), forecast evaluation (#137) and update impact (#138).

---

## 1. Data preparation (#131)

Code: `scripts/15_time_series_data.py` (about 10 seconds). Tables in `data/timeseries/`, charts in `data/charts/timeseries/`, numbers in `data/timeseries_summary.json`.

### 1.1 The series

The modelled series is the **daily count of 1–2★ reviews per app** (173 days, 1 Apr – 20 Sep 2026). Counts are used instead of the share of 1–2★ reviews, because the late-April feed gap removes mostly positive reviews, which distorts shares far more than counts.

| File | Contents |
|---|---|
| `daily_app.csv` | One row per app per day: reviews, 1–2★ and 4–5★ counts, mean rating, mean sentiment, share of 1–2★, the 9 issue counts, weekday, and the quality flags below |
| `daily_domain.csv` | The same, summed per domain |
| `weekly_app.csv`, `weekly_domain.csv` | Monday–Sunday weeks; `complete_week` marks weeks fully inside the window (24 of 25) |
| `release_events.csv` | One row per app version: first day seen, reviews, rating, share of 1–2★, and whether it counts as a new release |
| `stl_components.csv` | STL trend, weekly seasonal and remainder components per app |
| `patterns_by_app.csv` | Trend and seasonal strength, weekday effect, trend change, suggested differencing |
| `stationarity.csv` | ADF tests on the level, the first difference, the weekly difference and both |
| `acf_pacf.csv` | ACF and PACF for lags 0–28, for the level and the weekly difference |

### 1.2 Data-quality flags

| Flag | Meaning | How to use it |
|---|---|---|
| `feed_gap` | 21 Apr – 5 May for Swiggy, Blinkit, Domino's, Flipkart and Amazon (75 app-days). Positive reviews are mostly missing from the Play Store feed; 1–2★ counts fall far less | Pass it to ARIMA as an extra input (SARIMAX) so the model does not read the dip as a real change |
| `source_gap`, `source_coverage` | Zomato has no reviews from 23 Jul 22:00 to 25 Jul 12:30. Coverage is 0.92, 0 and 0.48 for those three days | 24 Jul is filled in `neg_reviews_filled` with the mean of the same weekday one and two weeks either side (165.5). 23 and 25 Jul keep their real counts, because reviews arrive in a burst after the outage |
| `neg_reviews_filled` | `neg_reviews` with only the fully missing Zomato day filled | Use this column for models; `neg_reviews` stays raw |

### 1.3 Release events

`app_version` is the version on the reviewer's phone, so the first day a version appears marks its rollout. 2,507 versions appear in the reviews; 558 have at least 30 reviews. A version counts as a **new release** (`use_for_update_impact`) only if it:

1. has at least 30 reviews;
2. first appears after 1 Apr (a version already present on 1 Apr was released earlier); and
3. is numbered higher than every version seen before it. This removes old builds that some users still run, which otherwise appear as "new" late in the window (for example Amazon 12.0 in September).

That leaves **224 release events**, from 12 (PhonePe) to 38 (Zomato) per app. `first_seen_10th_review` gives a date that ignores a few early-access reviews. Chart: `ts_04_release_events.png`.

### 1.4 Time-based patterns

**Levels and trend** (`ts_01_daily_negative_reviews.png`, `ts_02_stl_trend.png`). Flipkart (282 a day) and Blinkit (251) get the most 1–2★ reviews; Google Pay the fewest (27). Comparing the first and last four weeks (gap days excluded), Paytm rises most (+60%, a September jump) and Myntra (+22%), Amazon (+14%) and Blinkit (+13%) also rise, while Google Pay (−20%), Domino's (−18%), Swiggy (−16%) and Zomato (−11%) fall. The STL trend is strongest for Flipkart (strength 0.81), Paytm (0.67) and Myntra (0.60). The series also show short spikes, such as Swiggy in late July and Google Pay in mid-May, which a forecast should flag rather than predict.

**Weekly pattern** (`ts_03_weekday_effect.png`). Food apps get clearly more complaints on **Sundays**: Domino's +38%, Swiggy +28%, Zomato +28% and Blinkit +22% against their weekly mean, and fewer early in the week. Shopping apps have no real weekday pattern (within ±6%). Payment apps vary a little more: Google Pay +11% on Mondays and Thursdays and −11% on Saturdays, PhonePe −9% on Sundays. Seasonal strength from STL is 0.41–0.56 for Zomato and Blinkit and below 0.27 elsewhere.

**Stationarity** (`stationarity.csv`). The ADF test (lag length chosen by AIC) rejects a unit root at 5% for the level of 9 of 11 apps. Zomato (p = 0.066) and Blinkit (p = 0.094) are not stationary in level; after a first or weekly difference every app is stationary (p < 0.03).

**Autocorrelation** (`acf_pacf/<app>.png`, `acf_pacf.csv`). Lag-1 autocorrelation is 0.14 (Meesho) to 0.82 (Flipkart). Zomato and Blinkit have clear weekly spikes (ACF at lag 7: 0.43 and 0.50; at lag 14: 0.33 and 0.29); Flipkart, Myntra and Paytm show slow decay, which matches their stronger trend. The 95% limits are ±0.149.

### 1.5 Starting points for the ARIMA models (#135)

| | Suggestion | Why |
|---|---|---|
| Series | `neg_reviews_filled` per app | Raw counts with only the missing Zomato day filled |
| d (non-seasonal differencing) | 1 for Zomato and Blinkit, 0 for the rest (`suggested_d`) | ADF on the level |
| D (seasonal differencing) | 0 for every app (`suggested_D`) | STL seasonal strength is below 0.64 everywhere |
| Seasonal terms | Try a seasonal AR or MA term at lag 7 for Zomato, Blinkit and the other food apps | Weekly ACF spikes and the Sunday effect |
| Extra input | `feed_gap` as an exogenous dummy | The late-April dip for five apps |

These are starting points; the orders are chosen in #135 from the ACF/PACF charts and AIC.
