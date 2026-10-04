# EDA notes for Review 2

Moved out of `EDA.md` §7 ("Implications for other stages"). These notes come from the Review 1 EDA on the current dataset (1,218,358 reviews of 11 apps) and are meant for the Review 2 time-series and dashboard work. Paths are relative to the repository root.

**Person 5 — Time series & dashboard**
* Forecast **counts** of negative reviews per day, or exclude/flag 21 Apr – 5 May. The `overlaps_gap` flag is in `data/eda/weekly_trend.csv` and `contains_gap_days` is in `monthly_trend.csv`.
* All 11 apps share the same 25 complete weeks; no common-window logic is needed.
* `data/eda/version_metrics.csv` is a shortlist of unusual builds for release-impact analysis.
