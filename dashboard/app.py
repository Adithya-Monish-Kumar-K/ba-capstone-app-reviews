"""Review 2 dashboard — Mobile App Update Impact and Failure Detection.

Run from the repository root:   streamlit run dashboard/app.py

Each page lives in dashboard/pages/ and is registered below. The sidebar filters are drawn here once
and every page reads them with common.get_filters().
"""
import streamlit as st

from common import sidebar_filters

st.set_page_config(page_title="App Review Analytics", page_icon=":material/insights:", layout="wide")

PAGES = {
    "Overview": [
        st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    ],
    # Text mining:            st.Page("pages/text_mining.py", ...)            — #139
    # Forecasting, update impact and recommendations: st.Page("pages/...", ...) — #140
    # Check a review:         st.Page("pages/check_review.py", ...)           — #146
}

page = st.navigation(PAGES)
sidebar_filters()
page.run()
