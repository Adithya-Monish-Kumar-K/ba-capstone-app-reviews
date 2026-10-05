# Review 1 — Individual Contribution Summary

**Project:** Mobile App Update Impact and Failure Detection · **Course:** 23CSE452 Business Analytics · **Team:** 10

- **Project board (GitHub Projects):** https://github.com/users/Adithya-Monish-Kumar-K/projects/4
- **Repository:** https://github.com/Adithya-Monish-Kumar-K/ba-capstone-app-reviews
- **Board stages:** Backlog → To Do → In Progress → Review/Testing → Completed (15 tasks: #53, #54, #56–#58, #104–#110 and #118–#120, each with one or two assignees and a due date; every task is covered by a merged pull request: 10 are linked to their issue, and PR #121's description names the other five EDA issues)

| Member | Register number | Stage owned | Board tasks (issues) | Due | Evidence (merged PRs and files) |
|---|---|---|---|---|---|
| Aditya Monish Kumar K | CB.SC.U4CSE23103 | Data collection and preprocessing; project board | #53, #54, #56–#58 (5): scraper, raw collection, non-English filter and derived fields, cleaned dataset, data-integrity tests | 29 Sep | PR #52, #99 · `scripts/01_scrape_reviews.py`, `scripts/02_clean_reviews.py`, `scripts/apps.py`, `scripts/data_io.py`, `data/raw/`, `data/clean/`, `DATA_SOURCES.md` |
| Regella Krishna Saketh | CB.SC.U4CSE23649 | Exploratory data analysis (with Akshay KS) | #104–#106, #118–#120 (6): EDA script and charts, per-app and length breakdowns, coverage and bias audit, issue analysis, results write-up | 1 Oct | PR #121 (with Akshay KS) · `scripts/05_eda.py`, `data/charts/eda/`, `data/eda_summary.json`, `EDA.md` |
| Akshay KS | CB.SC.U4CSE23104 | Exploratory data analysis (with Regella Krishna Saketh) | #104–#106, #118–#120 (6): EDA script and charts, per-app and length breakdowns, coverage and bias audit, issue analysis, results write-up | 1 Oct | PR #121 · `scripts/05_eda.py`, `data/charts/eda/`, `data/eda_summary.json`, `EDA.md` |
| Harshini Vennela | CB.SC.U4CSE23455 | Feature engineering and predictive model (with Kanishka D) | #107–#110 (4): TF-IDF + SVD feature pipeline, Logistic Regression and Random Forest, evaluation charts, results write-up | 1 Oct | PR #122–#125 · `scripts/05_feature_engineering.py`, `scripts/06`–`13`, `models/`, `figures/`, `MODEL_EVALUATION.md` |
| Kanishka D | CB.SC.U4CSE23155 | Feature engineering and predictive model (with Harshini Vennela) | #107–#110 (4): TF-IDF + SVD feature pipeline, Logistic Regression and Random Forest, evaluation charts, results write-up | 1 Oct | PR #122–#125 (with Harshini Vennela) · `scripts/05_feature_engineering.py`, `scripts/06`–`13`, `models/`, `figures/`, `MODEL_EVALUATION.md` |

## Activity history (pull-request merges)

| Date (2026) | Activity |
|---|---|
| 28 Sep | 12 issues created (#53, #54, #56–#58, #104–#110); data collection and cleaning merged (PR #52, #99) |
| 29 Sep | GitHub Projects board created |
| 30 Sep | 3 EDA issues added (#118–#120); EDA merged (PR #121, Akshay KS and Regella Krishna Saketh); feature engineering, classifiers, evaluation and write-up merged (PR #122–#125, Harshini Vennela and Kanishka D) |

## Review 1 deliverables

| Rubric deliverable | File |
|---|---|
| Jupyter notebook | `notebooks/Review_1_Analysis.ipynb` (executed; HTML copy `Review_1_Analysis.html`) |
| Dataset documentation | `DATA_SOURCES.md` |
| Preprocessing, exploratory analysis and visualizations | Notebook §3–4, `EDA.md`, `data/charts/eda/` (15 charts) |
| Predictive model, evaluation and initial business findings | Notebook §5–7, `MODEL_EVALUATION.md`, `REPORT.md` §5–7 |
| Team presentation | `Review_1_Presentation.pptx` (PDF copy included) |
