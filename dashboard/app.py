"""
Interactive dashboard: Mobile App Update Impact & Failure Detection (Person 5, issue #112).

Run from the repository root:
    streamlit run dashboard/app.py

The dashboard reads the outputs already produced by the pipeline (see dashboard/utils.py for the list).
It does not re-run tagging, sentiment scoring, model training or forecasting.
Sidebar filters (app, period, metric) drive every tab; the Forecast tab uses the fixed window the
forecasts were fitted on, so only the app filter applies there.
"""
import pandas as pd
import streamlit as st

import charts
import utils as U

st.set_page_config(page_title="App Review Analysis Dashboard", page_icon="📊", layout="wide")

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def show(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


def section(title, help_text=None):
    st.subheader(title)
    if help_text:
        st.caption(help_text)


# ----------------------------------------------------------------------------
# Load existing outputs
# ----------------------------------------------------------------------------
reviews = U.load_reviews()
weekly = U.load_weekly()
forecast = U.load_csv("forecast", parse_dates=("week_start", "last_observed_week"))
accuracy = U.load_csv("accuracy")
backtest = U.load_csv("backtest", parse_dates=("origin_week", "target_week"))
ts_summary = U.load_json("ts_summary")
eda_summary = U.load_json("eda_summary")
sentiment_summary = U.load_json("sentiment_summary")
app_metadata = U.load_csv("app_metadata")
model_results = U.load_csv("model_results")
prediction_errors = U.load_csv("prediction_errors")

st.title("Mobile App Review Analysis")
st.caption("Play Store reviews of Swiggy, Zomato, Myntra, Paytm and PhonePe · issue tagging, sentiment, weekly trends and forecasts")

if reviews is None:
    st.error("`data/app_reviews_tagged.csv` was not found. Run the pipeline scripts (see README.md) and reload this page.")
    st.stop()

regime_week = ts_summary.get("weekly_aggregation", {}).get("first_post_shift_week")
final_model = ts_summary.get("forecast_setup", {}).get("final_model", "ses")

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
presets, first_date, last_date = U.preset_ranges(reviews, ts_summary)
with st.sidebar:
    st.header("Filters")
    app = st.selectbox("App", [U.OVERALL] + U.APPS, key="app",
                       help="Applies to every tab. 'All apps' pools the five apps.")
    period = st.radio("Period", list(presets) + ["Custom range"], key="period",
                      help="The common window is the 21 complete weeks in which every app has at least 30 reviews.")
    if period == "Custom range":
        default = next(iter(presets.values()))
        picked = st.date_input("Date range", value=default, min_value=first_date, max_value=last_date, key="dates")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = picked
        else:
            start = picked[0] if isinstance(picked, (tuple, list)) and picked else first_date
            end = last_date
            st.info("Pick an end date to complete the range.")
    else:
        start, end = presets[period]
    metric = st.selectbox("Metric", list(U.METRICS), format_func=lambda m: U.METRICS[m]["label"], index=2, key="metric",
                          help="Drives the app comparison on the Overview tab and the main chart on the Weekly trends tab.")
    hide_small = st.checkbox(f"Hide weekly rates based on fewer than {U.MIN_WEEK_N} reviews", value=True, key="hide_small",
                             help="Rates from very small weeks are noise. Review volume is always shown.")
    st.divider()
    st.caption(f"Showing **{app}**, {start:%d %b %Y} – {end:%d %b %Y}.")
    st.caption("Reviews come from Play Store's *most relevant* feed, which over-represents complaints. "
               "Compare apps and weeks with each other; do not read the levels as each app's true rating.")

sel = U.filter_reviews(reviews, app, start, end)          # selected app + period
period_all = U.filter_reviews(reviews, U.OVERALL, start, end)   # all apps, same period
k = U.kpis(sel)

tab_overview, tab_trends, tab_forecast, tab_text, tab_findings = st.tabs(
    ["Overview", "Weekly trends", "Forecast", "Sentiment & issues", "Key findings"])

# ----------------------------------------------------------------------------
# 1. Overview
# ----------------------------------------------------------------------------
with tab_overview:
    if k is None:
        st.warning(f"No reviews for **{app}** between {start:%d %b %Y} and {end:%d %b %Y}. Widen the period or pick another app.")
    else:
        ref = U.kpis(period_all) if app != U.OVERALL else None   # single app: show the gap to all apps in the same period

        def delta(name, fmt, unit=""):
            return None if ref is None else f"{k[name] - ref[name]:{fmt}}{unit} vs all apps"

        c = st.columns(4)
        c[0].metric("Total reviews", f"{k['review_count']:,}")
        c[1].metric("Apps", k["n_apps"])
        c[2].metric("First review in selection", f"{k['first']:%d %b %Y}")
        c[3].metric("Last review in selection", f"{k['last']:%d %b %Y}")
        c = st.columns(4)
        c[0].metric("Average rating", f"{k['mean_rating']:.2f}★", delta("mean_rating", "+.2f"))
        c[1].metric("Negative reviews (1–2★)", f"{k['pct_low_star']:.1f}%", delta("pct_low_star", "+.1f", " pp"), delta_color="inverse")
        c[2].metric("Positive reviews (4–5★)", f"{k['pct_high_star']:.1f}%", delta("pct_high_star", "+.1f", " pp"))
        c[3].metric("Negative sentiment (VADER)", f"{k['pct_negative_sentiment']:.1f}%",
                    delta("pct_negative_sentiment", "+.1f", " pp"), delta_color="inverse")
        c = st.columns(4)
        c[0].metric("With at least one issue tag", f"{k['pct_has_issue']:.1f}%", delta("pct_has_issue", "+.1f", " pp"), delta_color="inverse")
        c[1].metric("Median review length", f"{k['median_length']:.0f} words")
        c[2].metric("Mean sentiment score", f"{k['mean_sentiment']:+.3f}")
        if weekly is not None:
            n_weeks = U.filter_weekly(weekly, app, start, end, hide_small=False)["review_count"].gt(0).sum()
            c[3].metric("Weeks with reviews", int(n_weeks))

        st.divider()
        section(f"App comparison: {U.METRICS[metric]['label'].lower()}",
                "Value over the whole selected period. Change the metric or period in the sidebar; the selected app is highlighted.")
        left, right = st.columns([1, 1.25])
        with left:
            show(charts.app_comparison_bar(U.metric_by_app(period_all, metric), metric, app), "overview_compare")
        with right:
            st.dataframe(U.app_summary_table(period_all), hide_index=True, width="stretch", column_config={
                "Reviews": st.column_config.NumberColumn(format="localized"),
                "Average rating": st.column_config.NumberColumn(format="%.2f"),
                "% rated 1–2★": st.column_config.NumberColumn(format="%.1f"),
                "% rated 4–5★": st.column_config.NumberColumn(format="%.1f"),
                "% negative sentiment": st.column_config.NumberColumn(format="%.1f"),
                "% with issue tag": st.column_config.NumberColumn(format="%.1f"),
                "Median words": st.column_config.NumberColumn(format="%.0f"),
            })

        # ---- App-level exploration (only when a single app is selected) ----
        if app != U.OVERALL:
            st.divider()
            section(f"{app} profile", "Everything below follows the selected app and period.")
            c = st.columns(4)
            if app_metadata is not None:
                meta = app_metadata[app_metadata["app_name"].str.startswith(app)]
                if len(meta):
                    m = meta.iloc[0]
                    c[0].metric("Public Play Store rating", f"{m['current_score']:.2f}★")
                    c[1].metric("Rating in this sample", f"{k['mean_rating']:.2f}★", f"{k['mean_rating'] - m['current_score']:+.2f} vs public",
                                delta_color="off")
                    c[2].metric("Category", m["category"])
                    c[3].metric("Installs", m["installs"])
            if weekly is not None:
                prof = U.filter_weekly(weekly, app, start, end, hide_small, metrics=["mean_rating", "pct_low_star"])
                if prof["review_count"].sum() == 0:
                    st.info("No weekly data for this app in the selected period.")
                else:
                    c = st.columns(3)
                    for col, m_name, title in zip(c, ["review_count", "mean_rating", "pct_low_star"],
                                                  ["Review trend", "Rating trend", "Negative-review trend (1–2★)"]):
                        with col:
                            st.markdown(f"**{title}**")
                            if prof[m_name].notna().any():
                                show(charts.mini_trend(prof, m_name, U.APP_COLORS[app]), f"profile_{m_name}")
                            else:
                                st.info(f"No week has {U.MIN_WEEK_N}+ reviews in this period.")
            left, right = st.columns(2)
            with left:
                st.markdown("**Top issues reported**")
                show(charts.issue_bar(U.issue_table(sel).nlargest(5, "pct"), "pct", app, height=280), "profile_issues")
            with right:
                st.markdown("**4-week forecast, % rated 1–2★**")
                if forecast is not None and weekly is not None and len(forecast[forecast["series"] == app]):
                    hist, fc, bt = U.forecast_view(weekly, forecast, backtest, app, "pct_low_star", final_model, 1)
                    show(charts.forecast_figure(hist, fc, bt, "pct_low_star", app, final_model, 1, height=280, legend=False), "profile_forecast")
                    st.caption("Dashed line and band: forecast with 80% interval. Open circles: hold-out forecasts. "
                               "See the Forecast tab for other metrics, models and accuracy.")
                else:
                    st.info("No forecast is available for this app.")

    with st.expander("Data behind this dashboard"):
        st.dataframe(U.data_status(), hide_index=True, width="stretch")
        st.caption("All files are produced by the scripts in `scripts/`. The dashboard reads them; it does not recompute the analysis.")

# ----------------------------------------------------------------------------
# 2. Weekly trends
# ----------------------------------------------------------------------------
with tab_trends:
    if weekly is None:
        st.warning("Weekly tables not found. Run `python scripts/13_time_series_forecast.py` to create `data/timeseries/`.")
    else:
        section(f"{U.METRICS[metric]['label']} by week",
                "Weeks run Monday–Sunday. Hover for exact values. Change the app, period or metric in the sidebar.")
        if app == U.OVERALL:
            view = st.radio("Show", ["Each app", "All apps pooled", "Both"], horizontal=True, key="trend_view")
            names = {"Each app": U.APPS, "All apps pooled": [U.OVERALL], "Both": U.APPS + [U.OVERALL]}[view]
        else:
            names = [app] + ([U.OVERALL] if st.checkbox("Compare with all apps pooled", key="trend_compare") else [])
        rows = U.filter_weekly(weekly, names, start, end, hide_small, metrics=[metric])
        if rows.empty or rows[metric].notna().sum() == 0:
            st.info("No weekly values to plot for this selection. Widen the period, or untick the small-week filter in the sidebar.")
        else:
            show(charts.trend_figure(rows, metric, regime_week), "trend_main")
            notes = []
            if hide_small and U.METRICS[metric]["is_rate"]:
                hidden = int((rows[metric].isna() & (rows["review_count"] > 0)).sum())
                if hidden:
                    notes.append(f"{hidden} week(s) with fewer than {U.MIN_WEEK_N} reviews are hidden.")
            if rows["partial_week"].any():
                notes.append("Hollow marker = partial last week (scraped mid-week).")
            if metric == "review_count":
                notes.append("Each app was scraped to a fixed 3,000-review quota, so volume shows where the feed drew its reviews, not demand.")
            if notes:
                st.caption(" ".join(notes))

        section(f"Rating mix by week: {app}", "Share of each week's reviews rated 1–2★, 3★ and 4–5★.")
        mix = U.filter_weekly(weekly, app, start, end, hide_small, metrics=list(U.RATING_BANDS))
        mix = mix[mix["pct_low_star"].notna()]
        if mix.empty:
            st.info("No week in this selection has enough reviews to show a rating mix.")
        else:
            show(charts.rating_mix_figure(mix), "trend_mix")

        with st.expander("Weekly data table"):
            cols = ["series", "week_start", "week_end", "review_count", "mean_rating", "pct_low_star", "pct_high_star",
                    "pct_negative_sentiment", "pct_has_issue", "median_length", "partial_week"]
            table = U.filter_weekly(weekly, names, start, end, hide_small=False)[cols]
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "week_start": st.column_config.DateColumn("Week start"), "week_end": st.column_config.DateColumn("Week end")})
            st.download_button("Download as CSV", table.to_csv(index=False), file_name="weekly_metrics_selection.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# 3. Forecast
# ----------------------------------------------------------------------------
with tab_forecast:
    if forecast is None or weekly is None or accuracy is None:
        st.warning("Forecast outputs not found. Run `python scripts/13_time_series_forecast.py` to create `data/timeseries/`.")
    else:
        setup = ts_summary.get("forecast_setup", {})
        section(f"4-week forecast: {app}",
                "Forecasts were fitted once by the pipeline on the common window, so the period filter does not apply on this tab. "
                "The app filter does.")
        c = st.columns([1.4, 1.4, 1])
        f_metric = c[0].selectbox("Forecast metric", U.FORECAST_METRICS, format_func=lambda m: U.METRICS[m]["label"], key="f_metric")
        f_model = c[1].selectbox("Hold-out forecasts to overlay", list(U.MODEL_LABELS), index=list(U.MODEL_LABELS).index(final_model),
                                 format_func=lambda m: U.MODEL_LABELS[m], key="f_model",
                                 help="How each baseline model would have forecast the last 6 weeks, using earlier weeks only.")
        f_horizon = c[2].selectbox("Weeks ahead", [1, 2, 3, 4], key="f_horizon")
        hist, fc, bt = U.forecast_view(weekly, forecast, backtest, app, f_metric, f_model, f_horizon)
        if fc.empty or hist.empty:
            st.info(f"No forecast is available for {app} on this metric.")
        else:
            acc = U.accuracy_view(accuracy, app, f_metric)
            final = acc[acc["model"] == final_model]
            unit = "★" if f_metric == "mean_rating" else " pp"
            c = st.columns([1, 1, 1.5, 1.2])
            c[0].metric("Forecast, next 4 weeks", U.fmt_value(fc["forecast"].iloc[0], f_metric))
            c[1].metric("Last observed week", U.fmt_value(fc["last_observed"].iloc[0], f_metric))
            c[2].metric("80% interval at week 4", f"{fc['pi80_lower'].iloc[-1]:{U.METRICS[f_metric]['fmt']}} – "
                                                  f"{U.fmt_value(fc['pi80_upper'].iloc[-1], f_metric)}")
            if len(final):
                c[3].metric("Hold-out error (1–4 weeks)", f"{final['mae_h1_4'].iloc[0]:.2f}{unit}")
            show(charts.forecast_figure(hist, fc, bt, f_metric, app, f_model, f_horizon), "forecast_main")
            if pd.notna(fc["ses_alpha"].iloc[0]):
                st.caption(f"Smoothing weight α = {fc['ses_alpha'].iloc[0]:.2f} (closer to 1 = follows the latest weeks more closely).")
            st.caption(f"Solid line: observed weeks. Dashed line and band: {U.MODEL_LABELS.get(final_model, final_model)} forecast with its "
                       "80% prediction interval. Dotted line: what the chosen model forecast for each hold-out week before seeing it. "
                       "The forecast is flat because 21 weeks support a level, not a trend or season.")

            left, right = st.columns([1.2, 1])
            with left:
                section("Hold-out accuracy by model", f"{app}, {U.METRICS[f_metric]['label'].lower()}. Lower is better; blue = the model used for the forecast.")
                measure = st.radio("Error measure", list(U.ERROR_MEASURES), format_func=lambda m: U.ERROR_MEASURES[m], key="f_measure")
                show(charts.accuracy_bar(acc, measure, f_metric, final_model), "forecast_acc")
            with right:
                section("Forecast values")
                st.dataframe(fc[["week_start", "horizon", "forecast", "pi80_lower", "pi80_upper"]], hide_index=True, width="stretch",
                             column_config={"week_start": st.column_config.DateColumn("Week starting"),
                                            "horizon": "Weeks ahead",
                                            "forecast": st.column_config.NumberColumn("Forecast", format="%.2f"),
                                            "pi80_lower": st.column_config.NumberColumn("80% lower", format="%.2f"),
                                            "pi80_upper": st.column_config.NumberColumn("80% upper", format="%.2f")})
                st.dataframe(acc[["Model", "mae_h1", "mae_h1_4", "rmse_h1_4", "bias_h1_4", "skill_vs_naive"]], hide_index=True, width="stretch",
                             column_config={"mae_h1": st.column_config.NumberColumn("MAE 1 wk", format="%.2f"),
                                            "mae_h1_4": st.column_config.NumberColumn("MAE 1–4 wk", format="%.2f"),
                                            "rmse_h1_4": st.column_config.NumberColumn("RMSE", format="%.2f"),
                                            "bias_h1_4": st.column_config.NumberColumn("Bias", format="%+.2f"),
                                            "skill_vs_naive": st.column_config.NumberColumn("Skill vs naive", format="%+.2f")})

            section("Accuracy across all apps", f"{U.ERROR_MEASURES[measure]} for {U.METRICS[f_metric]['label'].lower()}, every series and model.")
            show(charts.accuracy_heatmap(accuracy, f_metric, measure), "forecast_heat")
            if setup:
                st.caption(f"Hold-out: the last {setup.get('test_weeks')} complete weeks ({setup.get('test_window', ['', ''])[0]} to "
                           f"{setup.get('test_window', ['', ''])[1]}), each forecast 1–4 weeks ahead from earlier weeks only. "
                           "Review volume is not forecast: weekly counts reflect the scraping quota, not review arrivals.")
            with st.expander("Hold-out forecasts behind the dotted line"):
                st.dataframe(bt[["origin_week", "target_week", "n_train_weeks", "forecast", "actual", "error"]], hide_index=True, width="stretch",
                             column_config={"origin_week": st.column_config.DateColumn("Forecast made after week of"),
                                            "target_week": st.column_config.DateColumn("Week forecast"),
                                            "n_train_weeks": "Training weeks"})

# ----------------------------------------------------------------------------
# 4. Sentiment & issues (Text Mining outputs)
# ----------------------------------------------------------------------------
with tab_text:
    if k is None:
        st.warning(f"No reviews for **{app}** between {start:%d %b %Y} and {end:%d %b %Y}. Widen the period or pick another app.")
    else:
        st.caption(f"Based on {len(sel):,} reviews: {app}, {start:%d %b %Y} – {end:%d %b %Y}. "
                   "Sentiment labels are the VADER labels and issue tags are the 9-category tags produced by the Text Mining stage.")
        left, right = st.columns(2)
        with left:
            section("Sentiment distribution")
            if app == U.OVERALL:
                groups = [(a, period_all[period_all["app_name"] == a]) for a in U.APPS] + [(U.OVERALL, period_all)]
            else:
                groups = [(app, sel), (U.OVERALL, period_all)]
            show(charts.sentiment_figure(U.sentiment_distribution(sel, groups), height=330 if app == U.OVERALL else 220), "text_sentiment")
        with right:
            section("Star-rating distribution")
            show(charts.rating_figure(U.rating_distribution(sel), height=330 if app == U.OVERALL else 220), "text_rating")

        section("Issue categories", "A review can carry several tags, so shares do not add up to 100%.")
        left, right = st.columns([1, 1.2])
        with left:
            issue_measure = st.radio("Rank issues by", list(charts.ISSUE_MEASURES), format_func=lambda m: charts.ISSUE_MEASURES[m][0],
                                     key="issue_measure")
            reference = U.issue_table(period_all) if app != U.OVERALL else None
            show(charts.issue_bar(U.issue_table(sel), issue_measure, app, reference), "text_issue_bar")
        with right:
            st.markdown("**Issue mix by app** (% of each app's reviews in the selected period)")
            matrix, counts = U.issue_by_app(period_all)
            show(charts.issue_heatmap(matrix, counts, height=400), "text_issue_heat")
            st.caption("The heatmap always shows all five apps so the selected app can be compared with the others.")

        if weekly is not None:
            section(f"Issue trend by week: {app}")
            ranked = U.issue_table(sel).sort_values("pct", ascending=False)["issue"].tolist()
            chosen = st.multiselect("Issue categories (up to 5)", U.ISSUES, default=ranked[:3], format_func=lambda i: U.ISSUE_LABELS[i],
                                    max_selections=5, key="issue_pick")
            issue_rows = U.filter_weekly(weekly, app, start, end, hide_small=False)
            if hide_small:
                issue_rows = issue_rows[issue_rows["review_count"] >= U.MIN_WEEK_N]
            if not chosen:
                st.info("Select at least one issue category.")
            elif issue_rows.empty:
                st.info(f"No week in this selection has {U.MIN_WEEK_N}+ reviews. Widen the period or untick the small-week filter.")
            else:
                show(charts.issue_trend_figure(issue_rows, chosen, U.ISSUE_LABELS, regime_week), "text_issue_trend")

        left, right = st.columns([1, 1])
        with left:
            section("Does sentiment agree with the star rating?", "Mean VADER compound score for each star rating in the selection.")
            show(charts.sentiment_by_star_figure(U.sentiment_by_star(sel)), "text_star")
        with right:
            val = sentiment_summary.get("validation", {})
            if val:
                section("Validation on the full dataset", "From the Text Mining stage (`data/sentiment_summary.json`).")
                c = st.columns(3)
                c[0].metric("Pearson r", f"{val.get('pearson_correlation', float('nan')):.3f}")
                c[1].metric("Spearman ρ", f"{val.get('spearman_correlation', float('nan')):.3f}")
                c[2].metric("3-class agreement", f"{val.get('overall_alignment_accuracy', float('nan')):.1f}%")
                st.caption("Agreement = share of reviews where the VADER label matches the rating class (1–2★ negative, 3★ neutral, 4–5★ positive).")

# ----------------------------------------------------------------------------
# 5. Key findings (numbers read from the stage summaries; wording follows the stage documents)
# ----------------------------------------------------------------------------


def findings():
    """(stage, finding, source) tuples. A finding is skipped when its source numbers are missing."""
    out = []

    def add(stage, source, build):
        try:
            out.append((stage, build(), source))
        except (KeyError, IndexError, TypeError):
            pass

    e, t, s = eda_summary, ts_summary, sentiment_summary
    add("Data", "EDA.md §1", lambda: (
        f"The sample is strongly negative and not representative: {e['rating_distribution_pct']['All apps']['1']:.1f}% of reviews are 1★ and "
        f"{e['rating_distribution_pct']['All apps']['5']:.1f}% are 5★, while every app's sample mean sits "
        f"{min(v['gap'] for v in e['sample_vs_public_rating'].values()):.1f}–{max(v['gap'] for v in e['sample_vs_public_rating'].values()):.1f}★ "
        "below its public rating. Only relative comparisons (app vs app, week vs week) are safe."))
    add("Text mining", "TEXT_MINING.md §4.1", lambda: (
        f"VADER sentiment agrees with star ratings: Pearson r = {s['validation']['pearson_correlation']:.2f}, and the sentiment label matches "
        f"the rating class for {s['validation']['overall_alignment_accuracy']:.1f}% of reviews."))
    add("Text mining", "EDA.md §5", lambda: (
        f"Customer Support is the most common issue ({e['issue_priority']['customer_support']['prevalence_pct']:.1f}% of reviews, average "
        f"{e['issue_priority']['customer_support']['mean_score']:.2f}★). Order Quality is rarer "
        f"({e['issue_priority']['order_quality_fulfillment']['prevalence_pct']:.1f}%) but rated worst "
        f"({e['issue_priority']['order_quality_fulfillment']['mean_score']:.2f}★)."))
    add("EDA", "EDA.md §5", lambda: (
        f"Every app has its own issue fingerprint: Delivery Delay is {e['issue_prevalence_by_app_pct']['Swiggy']['Delivery Delay']:.0f}% at Swiggy "
        f"and {e['issue_prevalence_by_app_pct']['Zomato']['Delivery Delay']:.0f}% at Zomato but "
        f"only {e['issue_prevalence_by_app_pct']['PhonePe']['Delivery Delay']:.0f}–{e['issue_prevalence_by_app_pct']['Paytm']['Delivery Delay']:.0f}% at the "
        f"UPI apps; Cancellation & Return is {e['issue_prevalence_by_app_pct']['Myntra']['Cancellation & Return']:.0f}% at Myntra; "
        f"Crash & Stability is {e['issue_prevalence_by_app_pct']['Paytm']['Crash & Stability']:.0f}% at Paytm."))
    add("EDA", "EDA.md §6.1", lambda: (
        f"A sampling shift hits all five apps in July 2026: reviews per day rise {e['regime_shift']['reviews_per_day_ratio_post_vs_pre']:.1f}× and "
        "reviews get much shorter. Issue-tag rates appear to fall "
        f"{-max(e['regime_shift']['issue_rate_change_raw_pp'].values()):.0f}–{-min(e['regime_shift']['issue_rate_change_raw_pp'].values()):.0f} pp, but only "
        f"{-max(e['regime_shift']['issue_rate_change_length_adjusted_pp'].values()):.0f}–"
        f"{-min(e['regime_shift']['issue_rate_change_length_adjusted_pp'].values()):.0f} pp once review length is held constant. "
        "Most of the apparent improvement is a collection artefact."))
    add("EDA", "EDA.md §6.2", lambda: (
        f"Evidence for 'bad releases' is weak: {e['version_issue_summary']['flagged_versions']} versions rate significantly worse than their app's "
        f"mean, and none of them shows a rise in crash or UI complaints ({e['version_issue_summary']['operational_signals']} of "
        f"{e['version_issue_summary']['elevated_signals']} elevated signals are fulfilment or support issues)."))
    add("Time series", "TIME_SERIES.md §1", lambda: (
        f"Only {t['weekly_aggregation']['common_window_weeks']} weeks ({t['weekly_aggregation']['common_window'][0]} onward) have enough reviews for "
        f"every app. In them the pooled share of 1–2★ reviews steps from {t['trend_statistics']['All apps']['pct_low_star']['mean_pre_shift']:.1f}% to "
        f"{t['trend_statistics']['All apps']['pct_low_star']['mean_post_shift']:.1f}% at the July shift, with no significant trend in any app afterwards."))
    add("Time series", "TIME_SERIES.md §8.2", lambda: (
        f"Simple level models forecast the weekly 1–2★ share with {t['forecast_accuracy']['primary_metric_mae_final_model_min']:.2f}–"
        f"{t['forecast_accuracy']['primary_metric_mae_final_model_max']:.2f} pp hold-out error. Exponential smoothing beats the naive forecast in "
        f"{t['forecast_accuracy']['final_model_beats_naive_count']} of {t['forecast_accuracy']['n_series_metric']} series; the mean of all history, "
        f"which ignores the July step, is the worst model in {t['forecast_accuracy']['hist_mean_worst_count']}."))
    add("Time series", "TIME_SERIES.md §9", lambda: (
        f"No release-style failure spike is visible: the weekly spike scan flags {t['issue_spike_scan']['n_flagged']} of "
        f"{t['issue_spike_scan']['app_issue_weeks_tested']} app-issue-weeks at more than 3 standard errors above baseline, against "
        f"{t['issue_spike_scan']['expected_by_chance']} expected by chance."))
    if model_results is not None and len(model_results):
        mr = model_results.set_index("model")
        add("Predictive model", "MODEL_EVALUATION.md §4", lambda: (
            f"Both classifiers separate problematic (1–2★) reviews well on the held-out test set: Logistic Regression F1 "
            f"{mr.loc['logistic_regression', 'f1_score']:.3f} (precision {mr.loc['logistic_regression', 'precision']:.3f}), Random Forest F1 "
            f"{mr.loc['random_forest', 'f1_score']:.3f} (recall {mr.loc['random_forest', 'recall']:.3f}). The split is random, so the scores do not "
            "show how the models would perform on later weeks."))
    return out


with tab_findings:
    section("Key findings", "Summarised from the stage documents; the numbers are read from the summary files each stage saved.")
    items = findings()
    if not items:
        st.info("Summary files were not found, so no findings can be shown.")
    for stage, text, source in items:
        st.markdown(f"**{stage}.** {text}  \n<span style='color:{U.MUTED};font-size:0.85em'>Source: {source}</span>", unsafe_allow_html=True)

    if model_results is not None:
        section("Predictive model results", "From `data/model_results.csv` and `data/prediction_error_summary.csv` (2,998-review test set).")
        left, right = st.columns(2)
        pct = st.column_config.NumberColumn(format="%.3f")
        left.dataframe(model_results.assign(model=model_results["model"].str.replace("_", " ").str.title()), hide_index=True, width="stretch",
                       column_config={c_: pct for c_ in model_results.columns if c_ != "model"})
        if prediction_errors is not None:
            right.dataframe(prediction_errors, hide_index=True, width="stretch")

    section("How to read this dashboard")
    st.markdown(
        "- **Levels are biased.** Reviews come from the *most relevant* feed, which favours complaints. Compare apps and weeks; do not quote the rates as population figures.\n"
        "- **July 2026 is a sampling change, not a product change.** Drops in negative share and issue rate at that point mostly reflect shorter sampled reviews.\n"
        "- **Forecasts are short-horizon level estimates** and hold only while the sampling regime stays as it has been since July.\n"
        "- Full method and limitations: `TEXT_MINING.md`, `EDA.md`, `MODEL_EVALUATION.md`, `TIME_SERIES.md`.")
