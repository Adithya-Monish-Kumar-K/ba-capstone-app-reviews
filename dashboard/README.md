# Review 2 Dashboard

Interactive Streamlit dashboard for the Review 2 results.

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py          # run from the repository root; opens http://localhost:8501
```

The app reads only small precomputed tables (for example `data/timeseries/daily_app.csv` from `scripts/15_time_series_data.py`), so it starts in a few seconds and does not load the 1.1 million raw reviews.

## Structure

| File | Purpose |
|---|---|
| `app.py` | Entry point: page list (`st.navigation`) and the sidebar filters |
| `common.py` | Shared data loaders, the `Filters` object, `apply_filters()`, chart styling, feed-gap shading |
| `pages/overview.py` | Overview page (#133): headline numbers, trends per domain, share of 1–2★ reviews, top issues |
| `pages/text_mining.py` | Text-mining page (#139): topic explorer with example reviews, topics by app, topic share over time, distinctive terms per app. Reads `data/textmining/*.csv` and `data/topic_modelling_summary.json` |
| `screenshots/` | Screenshots used as evidence in pull requests and the report |
| `../.streamlit/config.toml` | Theme |

## Sidebar filters

Every page shares the same filters: **domain**, **app**, **date range**, and **exclude the 21 Apr – 5 May feed gap**. A page reads them with:

```python
from common import apply_filters, get_filters, require_selection
f = get_filters()            # f.domains, f.apps, f.start, f.end, f.exclude_feed_gap
require_selection(f)         # stops the page with a message if no app is selected
df = apply_filters(my_table, f)   # keeps the selected apps and dates (needs app_name and day columns)
```

## Adding a page

1. Create `dashboard/pages/<name>.py`. Start with `import streamlit as st` and `from common import ...`, then `st.title(...)`.
2. Register it in `PAGES` in `app.py`, for example:
   ```python
   "Text mining": [st.Page("pages/text_mining.py", title="Text mining", icon=":material/topic:")],
   ```
3. Load data with a function decorated with `@st.cache_data`, from small files in `data/`.
4. Use `style(fig)` from `common.py` on Plotly figures and `shade_feed_gap(fig)` on time charts.
5. Run the app, check every filter combination, and add a screenshot to `screenshots/`.

Planned pages: check a review (#146), forecasting, update impact and recommendations (#140).

## Public link (Streamlit Community Cloud)

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository (the repository must be public).
2. **Create app** → repository `Adithya-Monish-Kumar-K/ba-capstone-app-reviews`, branch `master`, main file `dashboard/app.py`.
3. Deploy. The app installs `requirements.txt` and redeploys automatically whenever `master` changes.
