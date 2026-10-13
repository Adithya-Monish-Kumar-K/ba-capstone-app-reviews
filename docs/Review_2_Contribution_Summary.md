# Review 2 — Individual Contribution Summary

**Project:** Mobile App Update Impact and Failure Detection · **Course:** 23CSE452 Business Analytics · **Team:** 10

- **Project board (GitHub Projects):** https://github.com/users/Adithya-Monish-Kumar-K/projects/4
- **Repository:** https://github.com/Adithya-Monish-Kumar-K/ba-capstone-app-reviews
- **Live dashboard:** https://team10-app-reviews.streamlit.app/
- **Board stages:** Backlog → To Do → In Progress → Review/Testing → Completed.
- **Board tasks:** 14 tasks (#131–#143 and #146). Each has one assignee and a due date of 11 Oct, each is covered by a merged pull request that names it, and all 14 are Completed.
- **Order of work:** a relay of five stages. Each stage started on the merged work of the stage before. The slide deck, final report and this summary (Stage 6) were done last by Aditya Monish Kumar K and are not board tasks.

| Member | Register number | Stage owned | Board tasks (issues) | Evidence (merged PRs and files) |
|---|---|---|---|---|
| Aditya Monish Kumar K | CB.SC.U4CSE23103 | Stage 1: data preparation for both methods and the dashboard base. Stage 6: deck, final report, contribution summary | #131 daily and weekly time series, release events, STL, ADF and ACF/PACF; #132 text preprocessing and document-term matrices; #133 dashboard skeleton, shared filters, overview page and public deployment (3) | PR #147, #148, #149 · `scripts/15_time_series_data.py`, `scripts/16_text_preprocessing.py`, `data/timeseries/`, `data/textmining/`, `dashboard/app.py`, `dashboard/common.py`, `dashboard/pages/overview.py`, `TIME_SERIES.md` §1, `TEXT_MINING.md` §1, `docs/Review_2_Presentation.pptx`, `FINAL_REPORT.md` |
| Regella Krishna Saketh | CB.SC.U4CSE23649 | Stage 2: text mining (Method 1) | #134 NMF topic models, distinctive terms per app, sentiment per topic, praise contrast; #136 coherence, comparison with the keyword tags, untagged reviews, topic trends and write-up (2) | PR #150, #151 · `scripts/17_topic_modelling.py`, `scripts/18_topic_evaluation.py`, `models/textmining/`, `data/textmining/` (topic tables), `data/charts/textmining/` (tm_02–tm_09), `TEXT_MINING.md` §2–3 |
| Akshay KS | CB.SC.U4CSE23104 | Stage 3: ARIMA forecasting (Method 2) and the text-mining dashboard page | #135 ARIMA/SARIMAX forecasts with rolling-origin backtest, seasonal-naive and Holt-Winters baselines, 4-week forecast; #139 text-mining dashboard page with example reviews (2) | PR #152, #153 · `scripts/19_forecasting.py`, `scripts/20_topic_examples.py`, `data/timeseries/` (forecast and backtest tables), `dashboard/pages/text_mining.py`, `TIME_SERIES.md` §2 |
| Harshini Vennela | CB.SC.U4CSE23455 | Stage 4: forecast evaluation (Method 2) and the check-a-review page | #137 MAE/RMSE/MASE, interval coverage, Ljung–Box residual checks, unusual-day alerts; #142 forecast write-up; #146 check-a-review dashboard page (3) | PR #154, #155, #156 · `scripts/20_forecast_evaluation.py`, `data/timeseries/` (evaluation tables), `data/charts/timeseries/` (ts_08–ts_10), `dashboard/pages/check_review.py`, `TIME_SERIES.md` §2.5 |
| Kanishka D | CB.SC.U4CSE23155 | Stage 5: update impact (Method 2), combined insights, three dashboard pages, final notebook | #138 before/after and new-vs-older tests for 224 releases; #141 priority table and 65 recommendations; #140 forecasting, update-impact and recommendations pages; #143 final Review 2 notebook (4) | PR #157, #158 · `scripts/21_update_impact.py`, `scripts/22_combined_insights.py`, `data/update_impact/`, `data/combined/`, `dashboard/pages/forecasting.py`, `update_impact.py`, `recommendations.py`, `TIME_SERIES.md` §3, `COMBINED_INSIGHTS.md`, `notebooks/Review_2.ipynb` |

## Activity history (2026, IST)

| Date | Activity |
|---|---|
| 7 Oct | 14 Review 2 issues created and added to the board, each with an assignee and a due date of 11 Oct. Stage 1 pull requests opened (PR #147–#149, Aditya Monish Kumar K) |
| 8 Oct | Stage 1 merged: time series, text preprocessing, dashboard skeleton (PR #147–#149). Stage 2 merged: topic models and their evaluation (PR #150, #151, Regella Krishna Saketh) |
| 9 Oct | Stage 3 merged: ARIMA forecasting and the text-mining page (PR #152, #153, Akshay KS) |
| 10 Oct | Stage 4 merged: forecast evaluation, forecast write-up and the check-a-review page (PR #154–#156, Harshini Vennela) |
| 11 Oct | Stage 5 merged: update impact, combined insights, three dashboard pages and the final notebook (PR #157, #158, Kanishka D) |
| After 11 Oct | Stage 6: slide deck, final report and this summary (Aditya Monish Kumar K) |

## Review 2 deliverables

| Rubric deliverable | File |
|---|---|
| Method 1, text mining: preparation, implementation, evaluation and interpretation | `TEXT_MINING.md`, `FINAL_REPORT.md` §3, notebook §2 |
| Method 2, time series: preparation, implementation, evaluation and interpretation | `TIME_SERIES.md`, `FINAL_REPORT.md` §4, notebook §3–4 |
| Combined business insights and recommendations | `COMBINED_INSIGHTS.md`, `data/combined/`, `FINAL_REPORT.md` §5 |
| Interactive dashboard | https://team10-app-reviews.streamlit.app/ · `dashboard/` (6 pages) |
| Final report | `FINAL_REPORT.md` (PDF copy `docs/Final_Report.pdf`) |
| Final Jupyter notebook | `notebooks/Review_2.ipynb` (executed; HTML copy `notebooks/Review_2.html`) |
| Team presentation | `docs/Review_2_Presentation.pptx` (PDF copy included) |
