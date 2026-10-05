# Review 2 prep

Review 2 material moved out of the Review 1 files. Everything here was built on the **first dataset** (15,000 `MOST_RELEVANT` reviews of 5 apps) and must be redone on the current dataset (1,137,987 reviews of 11 apps, in `../data/tagged/`) before Review 2.

| Item | What it is |
|---|---|
| `REPORT_Review_2_sections.md` | Former REPORT.md §8–11 (Text Mining, Time-Series, Combined Insights, Dashboard), plus Review 2 conclusion points and reference |
| `TEXT_MINING.md` | Text-mining write-up (first-dataset numbers). The tagging/sentiment scripts stay in `../scripts/03`–`04` because Review 1 uses them; their charts are in `../data/charts/` |
| `TIME_SERIES.md`, `scripts/13_time_series_forecast.py`, `data/timeseries/`, `data/charts/timeseries/`, `data/timeseries_summary.json` | Weekly time-series and forecasts; run with `python review_2_prep/scripts/13_time_series_forecast.py` from the repository root |
| `dashboard/` | Streamlit dashboard; needs updating for the new dataset and EDA summary before it runs |
| `data/v1_most_relevant/` | The first dataset (raw, clean, tagged) |
| `old_v1_documents/` | The first combined report (PDF) and slide deck |
| `EDA_notes_for_Review_2.md` | EDA guidance for the Review 2 time series and dashboard |
