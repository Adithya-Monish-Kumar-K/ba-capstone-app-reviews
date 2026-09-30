# Exploratory Data Analysis — Findings & Method

**Capstone Project — Stage 4: Exploratory Data Analysis**
*Owner: Person 3 · Input: `data/app_reviews_tagged.csv` (14,988 reviews × 28 columns) · Code: `scripts/05_eda.py`*

Outputs: 13 charts in `data/charts/eda/`, all quoted numbers in `data/eda_summary.json`, and two tables for Person 5 (`data/eda/monthly_trend.csv`, `data/eda/version_metrics.csv`).

---

## 1. Headline findings

1. **The sample is extremely negative and not representative.** 69.2% of reviews are 1★, only 17.5% are 5★; every app's sample mean sits 2.0–3.3★ below its public rating ([chart 02](data/charts/eda/eda_02_ratings_and_sample_bias.png)). Only *relative* comparisons (app vs app, month vs month) are safe.
2. **The apps cover different time periods.** Swiggy, Zomato and Myntra are ~100% 2026 reviews; only Paytm and PhonePe reach back to 2018 ([chart 01](data/charts/eda/eda_01_coverage_timeline.png)). Only **Apr–Sep 2026** has ≥50 reviews for all five apps, so all cross-app trends use that *common window*.
3. **A sampling regime shift hits all five apps in July 2026.** Reviews per day rise 2.9× and median review length falls 27–50% ([chart 11](data/charts/eda/eda_11_july_regime_shift.png)). Issue-tag rates appear to fall 7–16 pp, but only 1–5 pp once review length is held constant. **Most post-July "improvement" is a collection artefact, not a change in the apps.** This is the most important caveat for Persons 4 and 5.
4. **Customer Support is the biggest problem; Order Quality is the most damaging.** Support appears in 34.4% of reviews (1.8× the next issue, average 1.19★). Order Quality is rare (6.6%) but rated worst (1.13★) ([chart 05](data/charts/eda/eda_05_issue_priority.png)).
5. **Every app has its own failure fingerprint.** Delivery Delay is ~30% at Swiggy/Zomato but 1–2% at the UPI apps; Cancellation & Return is 40% at Myntra; Crash & Stability is 17% at Paytm vs 2–3% at Myntra/Zomato/Swiggy ([chart 06](data/charts/eda/eda_06_issue_by_app.png)). All nine issue-vs-app chi-square tests have p < 10⁻³⁷.
6. **The issue tagger has both false positives and blind spots.** 333 five-star reviews carry *Cancellation & Return*, 92% of them Myntra praising easy returns ([chart 08](data/charts/eda/eda_08_issue_by_rating.png)). Conversely 22% of 1–2★ reviews have no tag; the missed terms are Paytm security-scan alerts ("malicious", "detected"), QR/scan, notifications and rewards/coins ([chart 09](data/charts/eda/eda_09_taxonomy_gap.png)).
7. **Upvotes are winner-take-all and do not explain the negative skew.** Gini 0.97; the top 1% of reviews hold 85% of upvotes. 1★ is 69% of reviews but only 43% of upvotes ([chart 03](data/charts/eda/eda_03_length_and_upvotes.png)).
8. **Unhappy users write about twice as much** (median 59 words at 1★ vs 30 at 5★), and issue tags track length (Spearman 0.40): longer text has more chances to match a keyword ([charts 03, 04](data/charts/eda/eda_04_correlation_matrix.png)).
9. **"Bad release" evidence by app version is weak, and where present it is not about crashes.** 8 of 84 versions rate significantly worse than their app's mean, but ~2 would be flagged by chance and version is confounded with time ([chart 12](data/charts/eda/eda_12_app_versions.png)). In those 8 versions the elevated complaints are fulfilment and support (8 of 9 significant signals); none shows a crash/stability or UI increase, and 2 show no specific issue at all ([chart 13](data/charts/eda/eda_13_version_issue_mix.png)).

---

## 2. Method

| Choice | Why |
|---|---|
| Read the tagged dataset only (no re-tagging or re-scoring) | Reproducible and independent of Person 2's stage. |
| **Common window** = months where every app has ≥50 reviews (Apr–Sep 2026; Sep is partial, scraped 20 Sep) | Coverage is very uneven (§3); cross-app trends outside the window compare different eras. |
| Points/rates shown only where n ≥ 30 | Avoids plotting noise as trend. |
| Medians, log upvotes, Spearman and rank tests | `thumbs_up` is heavy-tailed (max 46,743); ratings are ordinal. |
| Direct standardisation on review-length bands | Separates a real change in an app from a change in *what was sampled* (§6.1). |

One fixed colour per app across all charts; red↔blue for star ratings.

---

## 3. Data quality & coverage (charts 01–02)

| Check | Result |
|---|---|
| Rows / columns | 14,988 / 28 |
| Duplicate `review_id` | 0 |
| Nulls | Only `app_version`: 449 (3.0%) |
| Date range | 2018-09-15 → 2026-09-18 |
| Zero-upvote reviews | 40.5% |
| Untagged reviews (`has_issue = 0`) | 34.6% |

| App | Reviews | First review | Months with data | Distinct versions |
|---|---:|---|---:|---:|
| Swiggy | 3,000 | 2025-08 (99.9% from 2026) | 10 | 26 |
| Zomato | 2,997 | 2026-03 | 7 | 39 |
| Myntra | 2,998 | 2024-11 (99.9% from 2026) | 8 | 24 |
| Paytm | 2,998 | 2018-09 | 95 | 202 |
| PhonePe | 2,995 | 2018-09 | 96 | 177 |

1★ share: Swiggy 90%, Zomato 84%, Paytm 65%, Myntra 54%, PhonePe 53%. Rating differs strongly by app (chi-square V = 0.19).

**Outliers (Tukey 1.5 × IQR fences).**

| Field | Outside fences | Max | 99th percentile | Decision |
|---|---|---:|---:|---|
| `thumbs_up` | 1,897 (12.7%) | 46,743 | 399 | Keep. They are genuine viral reviews, not errors; use medians, `log1p` and rank tests. |
| `review_length` (words) | 9 (0.06%) | 223 | 98 | Keep. Long but plausible. |
| `sentiment_compound` | 0 | 0.99 | 0.98 | Bounded score, nothing to treat. |

No rows were removed. Removing the `thumbs_up` outliers would delete exactly the most-seen reviews, so the analysis is made robust instead (medians, log scale, Spearman, Mann-Whitney).

> **Correction to `DATA_SOURCES.md`.** It says users upvote complaints more than praise. Here that is not what the data shows: 4★ reviews are 4.5% of reviews but 23.5% of upvotes, and 1★ is 43% of upvotes vs 69% of reviews. The negative skew is real, but Play's `MOST_RELEVANT` ranking evidently uses other signals; the stated mechanism should be softened.

---

## 4. Engagement & correlations (charts 03–04)

* **Length:** median words fall monotonically with rating (59 → 53 → 45 → 36 → 30); Spearman ρ(rating, length) = −0.32.
* **Upvotes:** Gini 0.97; top 1% = 85% of all upvotes; PhonePe's mean (120) is 60× its median (2). Upvotes correlate with length (ρ = 0.36) but not with rating (−0.04) or sentiment (−0.01, p = 0.12).
* **Correlation structure:** VADER compound ↔ rating ρ = 0.56 (reproduces Person 2's validation); issue_count ↔ rating ρ = −0.45; issue_count ↔ length ρ = 0.40.

---

## 5. Issue analysis (charts 05–09)

| Issue | % of reviews | Mean ★ with issue | % rated 1–2★ | Effect vs rest (rank-biserial) |
|---|---:|---:|---:|---:|
| Customer Support | 34.4 | 1.19 | 94.9 | 0.36 |
| Cancellation & Return | 19.5 | 1.56 | 85.7 | 0.16 |
| Delivery Delay | 15.4 | 1.23 | 93.9 | 0.26 |
| Payment & Refund | 15.2 | 1.27 | 93.0 | 0.25 |
| Pricing & Fraud | 9.4 | 1.30 | 91.2 | 0.21 |
| Order Quality | 6.6 | **1.13** | 96.8 | 0.27 |
| Crash & Stability | 6.3 | 1.71 | 79.7 | 0.06 |
| UI/UX & Update | 2.4 | **2.71** | 52.2 | −0.24 |
| Account / Login / OTP | 2.4 | 1.49 | 85.1 | 0.13 |

All nine differ significantly in rating from untagged reviews (p < 0.001). **UI/UX & Update is the only issue rated above average**: 29% of its reviews are 5★ because the tag also matches praise such as "very UI friendly", and 72% of tagged reviews are Paytm/PhonePe, where it also catches update-triggered security warnings.

**By app (chart 06).** Strongest app effects (Cramér's V): Cancellation & Return 0.36, Delivery Delay 0.34, Crash & Stability 0.24, Customer Support 0.22. Support is high everywhere (Swiggy 48%, Zomato 45%, Myntra 31%, Paytm 27%, PhonePe 22%).

**Co-occurrence (chart 07).** Highest lifts: Crash+UI/UX 3.3× (n = 74), Payment+Order Quality 2.1× (n = 321), Payment+Cancellation 1.9× (n = 858). UI/UX almost never co-occurs with Order Quality (lift 0.08) or Delivery (0.13): app-experience complaints and physical-fulfilment complaints are largely separate.

**Tagger quality (charts 08–09).** Tags rise with lower ratings (79% of 1★ vs 26% of 5★ carry a tag), but the 5★ *Cancellation & Return* cluster is polarity-blind, and untagged low-star reviews are shorter (median 42 vs 62 words). Tag prevalence is therefore a lower bound on complaints, slightly inflated for Myntra returns praise.

---

## 6. Trends & versions (charts 10–13)

### 6.1 The July 2026 regime shift (chart 11)

| App | Median words before → after | Rating Δ raw | Rating Δ length-adjusted | Issue-rate Δ raw (pp) | Issue-rate Δ adjusted (pp) |
|---|---|---:|---:|---:|---:|
| Swiggy | 73 → 53 | +0.11 | +0.02 | −7.1 | −2.7 |
| Zomato | 70 → 47 | −0.03 | −0.20 | −10.4 | −5.2 |
| Myntra | 66 → 45 | +0.41 | −0.08 | −15.8 | −2.9 |
| Paytm | 41 → 26 | +0.64 | +0.38 | −8.7 | −1.7 |
| PhonePe | 40 → 20 | +0.80 | +0.18 | −14.6 | −1.2 |

"Adjusted" standardises each app's before/after values to that app's own length mix (bands ≤20, 21–40, 41–70, 71+ words). Within the 41–70 and 71+ bands the issue-tag rate is identical before and after (75%/75%, 84%/84%). Two real exceptions remain: 21–40 word reviews are tagged more often after July (45% → 56%), and ≤20-word reviews are rated ~0.5★ higher. The likely cause is a change in what Play's `MOST_RELEVANT` feed returned from July; the data alone cannot confirm it.

**Consequence.** Raw month-over-month gains for Myntra/Paytm/PhonePe ([chart 10](data/charts/eda/eda_10_monthly_trend.png)) must not be read as product improvements. Only Paytm (+0.38★) and PhonePe (+0.18★) keep a positive length-adjusted change; Zomato's is negative (−0.20★). Swiggy and Zomato never exceed 1.6★ in any month.

### 6.2 App versions (charts 12–13)
84 versions have ≥30 reviews. Eight are significantly worse than their app's mean (|z| > 2): Swiggy 4.109.1, Zomato 19.7.3, Myntra 4.2605.21 / 4.2606.11 / 4.2607.43, Paytm 10.79.1 / 10.80.0, PhonePe 26.04.24.0. Eleven are significantly better, mostly the newest PhonePe/Paytm builds (post-July). With 84 tests ~2 false flags per direction are expected, and a review is attributed to the reviewer's installed version, not the release that caused the complaint. Treat `data/eda/version_metrics.csv` as a shortlist for Person 5, not proof.

**What kind of failure?** For the 8 worse-rated versions, chart 13 compares each issue's rate with the app's overall rate (significant cells only: |diff| > 2 SE and ≥ 3 pp). Zomato 19.7.3 stands out (delivery +13 pp, customer support +13 pp, order quality +7 pp); Myntra 4.2605.21 and 4.2606.11 show more cancellation/return and support complaints; Swiggy 4.109.1 shows +8 pp delivery. **No flagged version has a significant rise in Crash & Stability or UI/UX.** Together with the tagger's polarity issues this suggests that low-rated versions mostly coincide with operational problems (deliveries, support) rather than app defects, or that ratings dip for reasons the text does not name; we cannot separate the two from reviews alone.

---

## 7. Implications for other stages

**Person 4 — Predictive modelling**
* `is_problematic` (score ≤ 2) has a **74.1% base rate**; a majority-class model scores 74.1% accuracy. Report precision/recall/F1/PR-AUC, not accuracy.
* `issue_count` is partly a **length proxy** (ρ = 0.40). Include `review_length` explicitly and check the model is not only learning length.
* The July shift changes feature distributions. Split train/test by time carefully (a random split hides it), or add a `post_july` feature.

**Person 5 — Time series & dashboard**
* Use only the common window (Apr–Sep 2026) for cross-app series; Paytm/PhonePe long history is not like-for-like (pre-2024 rows are ~100 survivors per year).
* Do not forecast raw volume or tag rate across the July step without modelling a level shift; prefer length-standardised series.
* `data/eda/monthly_trend.csv` has per-app monthly metrics over all months (rating, low-star %, issue rates, sentiment, length) with `in_common_window`, `post_regime_shift`, `partial_month` and `n_ge_min` flags, so the common window and the July step can be handled without re-deriving them.
* `data/eda/version_metrics.csv` is a ready shortlist of unusual builds for release-impact analysis.

**Person 2 — Tagging (feedback)**: add polarity handling for *Cancellation & Return*, and patterns for security-scan alerts, QR/scan, notifications, rewards/coins.

---

## 8. Statistical tests (in `data/eda_summary.json`)

| Test | Result |
|---|---|
| Rating vs app (chi-square) | χ² = 2228, dof = 16, Cramér's V = 0.19 |
| Rating by app (Kruskal–Wallis) | H = 1834 |
| Issue vs app (9 chi-square tests) | all p < 10⁻³⁷; V from 0.11 to 0.36 |
| Rating with vs without each issue (Mann-Whitney) | all p < 0.001; largest effect Customer Support (0.36) |
| Upvotes, tagged vs untagged (Mann-Whitney) | rank-biserial −0.09 (tiny) |
| Upvotes vs length / sentiment (Spearman) | ρ = 0.36 / −0.01 (p = 0.12) |

With n ≈ 15,000 p-values are almost always tiny; read the **effect sizes** (V, ρ, rank-biserial).

---

## 9. Limitations

* Rating/sentiment levels are biased by `MOST_RELEVANT` sampling and are not population estimates.
* One snapshot (20 Sep 2026): the cause of the July shift cannot be established from this dataset.
* Review time has no stored timezone.
* Tags are rule-based: recall depends on length and some categories have polarity blind spots.
* `app_version` is the reviewer's installed version, missing for 3.0% of rows; PhonePe mixes two version schemes (`26.xx.xx.x` and `4.26xx.xx`).
* Comparisons are exploratory; multiple-comparison correction is discussed only for versions (§6.2).

---

## 10. Reproducing Stage 4

```bash
pip install pandas numpy scipy matplotlib scikit-learn
python scripts/05_eda.py    # ~10 s; writes 13 charts, eda_summary.json, eda/monthly_trend.csv, eda/version_metrics.csv
```
