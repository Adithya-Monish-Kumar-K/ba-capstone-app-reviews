# Exploratory Data Analysis — Findings & Method

**Capstone Project — Stage 4: Exploratory Data Analysis**
*Owners: Person 3 and Person 2 · Input: `data/tagged/*.csv.gz` (1,137,987 reviews × 29 columns, 11 apps) · Code: `scripts/05_eda.py`*

Outputs: 15 charts in `data/charts/eda/`, all quoted numbers in `data/eda_summary.json`, and three tables (`data/eda/monthly_trend.csv`, `data/eda/weekly_trend.csv`, `data/eda/version_metrics.csv`).

---

## 1. Headline findings

1. **Recent reviews are short and mostly positive.** 67% are 5★ and 17% are 1★. The median review is 2 words, and 66% have 3 words or fewer ([chart 02](data/charts/eda/eda_02_ratings_vs_public.png), [chart 03](data/charts/eda/eda_03_length_and_upvotes.png)).
2. **Amazon and Swiggy are the clear outliers.** 54% of Amazon reviews and 35% of Swiggy reviews are 1–2★, against 10–24% for every other app. PhonePe (9.8%) and Myntra (10.3%) are the mildest. Written reviews sit 0.0–1.4★ below each app's public rating; Amazon has the biggest gap (2.74 vs 4.18★).
3. **A late-April feed gap, not an incident.** From 21 April to 5 May 2026, positive reviews fall 59–90% for Swiggy, Blinkit, Domino's, Flipkart and Amazon, while negative reviews fall only 3–37% ([chart 15](data/charts/eda/eda_15_april_gap.png)). Swiggy's weekly "% rated 1–2★" jumps from ~33% to 87% mainly because short positive reviews ("good", "nice") are missing. **Shares are fragile; counts of negative reviews per day are far less affected.** The window is flagged and excluded from trend, version and July analyses.
4. **No mid-year shift.** Comparing July–September with April–June, no app gets ≥1.5× more reviews per day, and median length is unchanged in 10 of 11 apps ([chart 11](data/charts/eda/eda_11_july_check.png)). A time-based train/test split is therefore safe.
5. **Customer Support is both the most common and the most damaging issue.** It appears in 4.1% of all reviews, and those reviews average 1.23★ ([chart 05](data/charts/eda/eda_05_issue_priority.png)). Only 10% of reviews carry any issue tag, because most reviews are too short to name a problem.
6. **Each domain has its own fingerprint.** Customer Support leads Food & Grocery; Cancellation & Return leads Shopping. In Payments, Crash & Stability and Customer Support are tied (1.30% and 1.29%), and delivery problems are almost absent. Delivery Delay is the most domain-specific issue (3.5% of Food & Grocery reviews vs 0.07% of Payments) ([charts 06, 14](data/charts/eda/eda_14_domain_comparison.png)). Food & Grocery is the most negative domain (22% 1–2★), Payments the mildest (13%).
7. **Ratings are stable over time; app differences dominate.** Between May–Jun and Aug–Sep, 7 of 11 apps move by less than 2.5 pp of 1–2★ share. Exceptions: Amazon worsens by 9.6 pp, Zomato by 4.4 pp and Flipkart by 2.6 pp; Google Pay improves by 5.5 pp ([chart 10](data/charts/eda/eda_10_weekly_trend.png)).
8. **The tagger misses most short complaints, and many 1–2★ reviews are mis-ratings.** 54% of 1–2★ reviews get no tag. Their median length is 5 words, against 28 for tagged ones. The most distinctive untagged terms are praise ("nice product", "mast", "super", "gud"), which means users giving 1★ by mistake ([chart 09](data/charts/eda/eda_09_taxonomy_gap.png)).
9. **"Bad release" signals exist but are concentrated.** 49 of 544 versions are clearly worse than their app's mean (≥0.25★ below and z ≤ −3); 15 of them are Flipkart builds. Every Amazon version released since mid-July (five versions, 32.13 to 32.17) is flagged, matching its downward trend. In the 15 worst versions, the most often elevated issue is Customer Support (9 signals), followed by Payment & Refund (8) ([charts 12–13](data/charts/eda/eda_12_app_versions.png)).

---

## 2. Method

| Choice | Why |
|---|---|
| Read the tagged dataset only (no re-tagging or re-scoring) | Reproducible and independent of the tagging stage. |
| Identical window for every app (1 Apr – 20 Sep 2026) | `NEWEST` collection back to a fixed date, so no common-window restriction is needed. |
| Complete weeks / full months only for trends | The first week and September (1–20 Sep) are partial. |
| **Exclude 21 Apr – 5 May** from trend, version and July analyses | Positive reviews are largely missing for 5 apps in this window (§6.1). |
| Version flags need **≥0.25★ gap AND \|z\| ≥ 3** | With 1.1M rows a plain z ≥ 2 flags trivial differences. |
| Medians, log upvotes, Spearman and rank tests | Upvotes are heavy-tailed; ratings are ordinal. |
| Eleven apps are never drawn in one panel | Time charts are split by domain. Each app keeps one colour, taken from a palette checked for colour-blind readers (`scripts/apps.py`). |

---

## 3. Data quality & coverage (charts 01–02)

| Check | Result |
|---|---|
| Rows / columns | 1,137,987 / 29 |
| Duplicate `review_id` | 0 |
| Nulls | Only `app_version`: 152,017 (13.4%) |
| Date range | 2026-04-01 → 2026-09-20 (identical for every app) |
| Days with reviews | 173 of 173 for every app except Zomato (172): no reviews from 23 Jul 22:00 to 25 Jul 12:30 (24 July is empty); re-querying the Play Store returned the same counts, so the gap is in the source, not the scraper |
| Zero-upvote reviews | 93.6% |
| Reviews with ≤ 3 words | 65.6% |
| Untagged reviews (`has_issue = 0`) | 89.6% |

| App | Domain | Reviews | Per day | Mean ★ | % 1–2★ | Median words | Versions |
|---|---|---:|---:|---:|---:|---:|---:|
| Flipkart | Shopping | 266,285 | 1,539 | 4.07 | 18.3 | 2 | 184 |
| Blinkit | Food & Grocery | 228,446 | 1,320 | 4.07 | 19.0 | 2 | 221 |
| Zomato | Food & Grocery | 150,560 | 870 | 4.06 | 18.8 | 2 | 345 |
| Meesho | Shopping | 102,686 | 594 | 4.22 | 17.6 | 3 | 217 |
| PhonePe | Payments | 88,535 | 512 | 4.43 | 9.8 | 2 | 193 |
| Myntra | Shopping | 85,347 | 493 | 4.49 | 10.3 | 3 | 138 |
| Swiggy | Food & Grocery | 74,768 | 432 | 3.45 | 35.0 | 2 | 266 |
| Paytm | Payments | 44,658 | 258 | 4.31 | 14.6 | 2 | 180 |
| Domino's | Food & Grocery | 43,629 | 252 | 3.89 | 23.6 | 2 | 125 |
| Amazon | Shopping | 32,753 | 189 | 2.74 | 54.4 | 7 | 219 |
| Google Pay | Payments | 20,320 | 117 | 3.93 | 23.2 | 2 | 419 |

Rating differs by app (chi-square Cramér's V = 0.12) much more than by domain (V = 0.06).

**Outliers (Tukey 1.5 × IQR fences).**

| Field | Outside fences | Max | 99th percentile | Decision |
|---|---|---:|---:|---|
| `thumbs_up` | 72,945 (6.4%) | 11,165 | 4 | Keep. Any review with a vote is "outside" because 94% have none; use `log1p` and rank tests. |
| `review_length` (words) | 167,193 (14.7%) | 167 | 82 | Keep. The fence is only 11 words because the median is 2; long reviews are the informative ones. |
| `sentiment_compound` | 36,775 (3.2%) | 1.00 | 0.89 | Bounded score, nothing to treat. |

No rows were removed by the EDA.

---

## 4. Engagement & correlations (charts 03–04)

* **Length:** median words by rating are 14 / 4 / 2 / 2 / 2 for 1★ to 5★. A complaint is a sentence; praise is one word. Spearman ρ(rating, length) = −0.39.
* **Upvotes:** Gini 0.98. 93.6% of reviews have no upvote, and the top 1% hold 71% of all upvotes. 1★ reviews are 17% of reviews but 59% of upvotes: other users endorse complaints far more than praise.
* **Correlation structure:** rating ↔ VADER compound ρ = 0.49; rating ↔ issue count ρ = −0.50; issue count ↔ length ρ = 0.46. Longer text has more chances to match a keyword rule.

---

## 5. Issue analysis (charts 05–09, 14)

| Issue | % of reviews | Reviews | Mean ★ with issue | % rated 1–2★ | Effect vs rest (rank-biserial) |
|---|---:|---:|---:|---:|---:|
| Customer Support | 4.07 | 46,338 | 1.23 | 94.1 | 0.80 |
| Cancellation & Return | 3.09 | 35,112 | 1.50 | 87.0 | 0.73 |
| Delivery Delay | 2.81 | 32,030 | 1.57 | 84.7 | 0.71 |
| Payment & Refund | 1.65 | 18,726 | 1.38 | 90.0 | 0.75 |
| Pricing & Fraud | 1.64 | 18,674 | 1.42 | 87.7 | 0.75 |
| Order Quality | 0.91 | 10,335 | 1.24 | 93.4 | 0.78 |
| Crash & Stability | 0.53 | 6,034 | 1.78 | 77.6 | 0.66 |
| Account / Login / OTP | 0.19 | 2,127 | 1.45 | 87.6 | 0.72 |
| UI/UX & Update | 0.18 | 2,061 | 2.74 | 51.8 | 0.40 |

All nine differ significantly in rating from untagged reviews (p < 0.001). UI/UX & Update is again the mildest: the tag also matches praise such as "nice UI".

**By app (chart 06).** Amazon has the heaviest issue load in every category except Crash & Stability (Customer Support 16.8%, Cancellation & Return 12.5%, Delivery Delay 8.9%). Swiggy is second (Customer Support 9.9%, Delivery Delay 6.1%). Crash & Stability is highest at Google Pay (2.7%), Paytm (2.2%) and Amazon (2.0%); PhonePe is low (0.5%). Delivery Delay and Order Quality are close to zero at the payment apps. Strongest app effects (Cramér's V): Customer Support 0.146, Cancellation & Return 0.133, Delivery Delay 0.103.

**By domain (chart 14).** Shopping's top issue is Cancellation & Return (4.6%, just ahead of Customer Support at 4.4%). Food & Grocery is led by Customer Support (4.6%). In Payments, Crash & Stability (1.30%) and Customer Support (1.29%) are tied. Delivery Delay is about 50× more common in Food & Grocery (3.53%) than in Payments (0.07%).

**Co-occurrence (chart 07).** Crash + UI/UX co-occur 26× more than chance (n = 281), i.e. post-update breakage. Payment & Refund travels with Cancellation & Return (13.3×, n = 7,666) and Order Quality (13.1×, n = 2,236): a wrong or cancelled order becomes a refund complaint.

**Tagger quality (charts 08–09).** 49% of 1★ reviews carry a tag, against 1.3% of 5★. The largest 5★ false-positive cluster is Cancellation & Return (3,135 reviews, 40% Myntra; 97% of Myntra's mention returns or exchanges, i.e. praise for easy returns). 54% of 1–2★ reviews have no tag. They are short (median 5 words; 30% are 1–2 words), and the most distinctive terms are praise words ("nice product", "mast", "super", "gud", "badhiya") or unmodelled topics ("compatible" on Amazon, "currently unavailable" and "late service" on Blinkit). Tag prevalence is therefore a lower bound on complaints.

---

## 6. Trends, data gaps & versions (charts 10–13, 15)

### 6.1 The 21 April – 5 May feed gap (chart 15)

| App | Positive/day before → gap → after | Negative/day before → gap → after |
|---|---|---|
| Swiggy | 297 → **30** → 325 | 163 → 136 → 184 |
| Domino's | 180 → **31** → 214 | 72 → 47 → 68 |
| Flipkart | 1,254 → **162** → 1,556 | 257 → 178 → 309 |
| Blinkit | 966 → **414** → 1,198 | 226 → 213 → 290 |
| Amazon | 103 → **42** → 100 | 96 → 97 → 105 |
| Meesho (unaffected) | 491 → **446** → 452 | 102 → 103 → 100 |

Before = 8–20 Apr, after = 6–18 May. An app is "affected" when positive reviews per day fall below 60% of normal *and* fall much more than negative ones. Five apps meet the rule; Paytm is borderline (positive 54% of normal, negative 79%). Median review length jumps in the affected apps (Swiggy 2 → 18 words), so the missing reviews are the short positive ones. A scraper failure would remove all reviews alike, so the cause is on the Play Store side; this dataset cannot say what it was.

**Consequence.** Any share-based metric ("% negative", mean rating) is distorted in this window. Counts of negative reviews per day are far less affected (−3% to −37% in the affected apps, and they did not rise). 61,077 reviews fall in the window; they are kept and flagged.

### 6.2 Weekly trends (chart 10)

Change in % rated 1–2★ from May–Jun to Aug–Sep (gap excluded; Aug–Sep runs to 20 Sep): Amazon +9.6 pp, Zomato +4.4, Flipkart +2.6, Myntra +2.0, Swiggy +1.1, Meesho +0.4, Blinkit +0.4, PhonePe −0.1, Paytm −0.3, Domino's −1.1, Google Pay −5.5. Amazon's mean rating falls 0.37★ over the same period. The most volatile apps week to week are Amazon (SD 0.21★) and Google Pay (0.18★); the most stable is Blinkit (0.03★). Differences between apps (Cramér's V = 0.12) are far larger than any app's movement.

### 6.3 The July 2026 check (chart 11)

This check asks whether the review mix changes partway through the window, which would make a time-based test unreliable. Reviews per day in July–September (to 20 Sep) are 0.62–1.02× the April–June level (gap days excluded): no app rises by more than 2% (Flipkart), and Zomato's volume actually falls by about 38%. Median length is unchanged in 10 of 11 apps; Amazon's goes up by 2 words. Length-adjusted rating changes are small (−0.19 Zomato to +0.13 Google Pay). **There is no common July shift.**

### 6.4 App versions (charts 12–13)

544 versions have ≥30 reviews (gap days excluded). 49 are clearly worse than their app's mean and 17 clearly better. Worse versions by app: Flipkart 15, Zomato 7, Swiggy 5, Amazon 5, PhonePe 5, Google Pay 4, Paytm 3, Blinkit 2, Meesho 2, Myntra 1. Amazon stands out: every version first seen since 15 July (32.13.0 to 32.17.0, five versions) rates 2.28–2.48★ against the app mean of 2.75★, consistent with Amazon's falling trend.

For the 15 most significantly worse versions, chart 13 compares each issue's rate with the app's overall rate (significant cells only: |diff| > 3 SE and ≥ 2 pp). Elevated signals: Customer Support 9, Payment & Refund 8, Delivery Delay 5, Cancellation & Return 4, Crash & Stability 3, Order Quality 2, Pricing & Fraud 1. One version shows no specific issue. Some versions do show crash increases. Version is still confounded with time, and a review is attributed to the reviewer's installed version, not the release that caused the complaint. Treat `data/eda/version_metrics.csv` as a shortlist, not proof.

---

## 7. Implications for other stages

**Person 4 — Predictive modelling**
* `is_problematic` (score ≤ 2) has a **19.5% base rate**. The classes are imbalanced: a model that never flags anything scores 80.5% accuracy. Report precision, recall, F1 and PR-AUC.
* Most reviews are 1–3 words, so text features carry little for them. Sentiment and length matter most. `issue_count` is partly a length proxy (ρ = 0.46).
* There is no shift partway through the window, so a time-based test split is safe. The April gap affects only the label mix in two weeks.

**Tagging stage (feedback)**: add polarity handling for *Cancellation & Return* (praise for easy returns), and patterns for compatibility, stock availability ("currently unavailable") and Hinglish complaint words.

---

## 8. Statistical tests (in `data/eda_summary.json`)

| Test | Result |
|---|---|
| Rating vs app (chi-square) | χ² = 65,055, dof = 40, Cramér's V = 0.12 |
| Rating vs domain (chi-square) | χ² = 7,469, dof = 8, Cramér's V = 0.06 |
| Rating by app (Kruskal–Wallis) | H = 44,490 |
| Issue vs app (9 chi-square tests) | all p < 0.001; V from 0.04 to 0.15 |
| Rating with vs without each issue (Mann-Whitney) | all p < 0.001; largest effect Customer Support (0.80) |
| Upvotes, tagged vs untagged (Mann-Whitney) | mean 1.32 vs 0.11; rank-biserial −0.22 |
| Review length, tagged vs untagged | median 26 vs 2 words |
| Upvotes vs length (Spearman) | ρ = 0.26 |

With n ≈ 1.1 million every p-value is tiny. Read the **effect sizes** (V, ρ, rank-biserial).

---

## 9. Limitations

* Written reviews are harsher than the public rating (which includes star-only ratings), so absolute levels are not population satisfaction.
* The cause of the late-April gap cannot be established from this dataset; we only measure its effect.
* One snapshot. Reviews edited or deleted before collection are not visible.
* Review time has no stored timezone.
* Tags are rule-based. Most reviews are too short to tag, and some categories have polarity blind spots.
* `app_version` is the reviewer's installed version and is missing for 13.4% of rows.

---

## 10. Reproducing Stage 4

```bash
pip install pandas numpy scipy matplotlib scikit-learn
python scripts/05_eda.py    # ~25 s; writes 15 charts, eda_summary.json and the three tables in data/eda/
```
