import re
import pandas as pd

df = pd.read_csv("/home/Adithya/Desktop/Sem 7/BA/BAProject/topic1_app_review_dataset/app_reviews_raw.csv")
df["review_date"] = pd.to_datetime(df["review_date"])
df["content"] = df["content"].fillna("")
df["month"] = df["review_date"].dt.to_period("M").astype(str)

ISSUE_KEYWORDS = {
    "crash":            r"\bcrash|\bcrashing|\bcrashes|\bforce close|\bnot opening|\bwon'?t open|\bfreez",
    "login_auth":       r"\blogin|\blog in|\bsign in|\botp\b|\bverification\b|\baccount blocked|\blogged out",
    "payment":          r"\bpayment fail|\bpayment issue|\btransaction fail|\bmoney deduct|\brefund|\bupi fail|\bcashback",
    "performance_slow": r"\bslow|\blag\b|\blagging|\bhang\b|\bhanging|\bbuffer|\btakes forever|\bloading forever",
    "bug_glitch":       r"\bbug\b|\bbugs\b|\bglitch|\bnot working|\bstopped working|\berror\b",
    "customer_support": r"\bcustomer (care|support|service)|\bno response|\bno reply|\bcomplaint ignored|\bsupport is (useless|bad|pathetic)",
    "ui_ux":            r"\binterface|\bui\b|\bux\b|\bconfusing|\bhard to use|\bredesign|\bnew update.{0,15}(bad|worse|ugly)",
    "ads_spam":         r"\bads\b|\badvertisement|\bspam\b|\bpop.?up",
    "delivery_order":   r"\bdelivery (late|delay)|\bdelayed order|\bwrong (item|order)|\bcancel(led)? order|\border not",
    "notifications":    r"\bnotification|\bspam(my)? notification",
}

for name, pattern in ISSUE_KEYWORDS.items():
    df[f"issue_{name}"] = df["content"].str.contains(pattern, case=False, regex=True, na=False)

df["any_issue_flagged"] = df[[f"issue_{k}" for k in ISSUE_KEYWORDS]].any(axis=1)

out_tagged = "/home/Adithya/Desktop/Sem 7/BA/BAProject/topic1_app_review_dataset/app_reviews_tagged.csv"
df.to_csv(out_tagged, index=False)
print("Saved tagged dataset:", len(df), "rows ->", out_tagged)
print(f"Reviews with at least one issue keyword matched: {df['any_issue_flagged'].mean():.1%}")

# ---- monthly aggregation per app: avg score + issue mention rate + dominant version ----
monthly_rows = []
for (app_name, month), g in df.groupby(["app_name", "month"]):
    if len(g) < 5:
        continue
    row = {
        "app_name": app_name,
        "month": month,
        "review_count": len(g),
        "avg_score": g["score"].mean(),
        "dominant_version": g["app_version"].mode().iloc[0] if not g["app_version"].dropna().empty else None,
    }
    for k in ISSUE_KEYWORDS:
        row[f"pct_{k}"] = g[f"issue_{k}"].mean()
    monthly_rows.append(row)

monthly = pd.DataFrame(monthly_rows).sort_values(["app_name", "month"])
out_monthly = "/home/Adithya/Desktop/Sem 7/BA/BAProject/topic1_app_review_dataset/monthly_trend.csv"
monthly.to_csv(out_monthly, index=False)
print("Saved monthly trend:", len(monthly), "rows ->", out_monthly)

# ---- Find the sharpest month-over-month rating drops per app, and what issue spiked alongside ----
print("\n=== Sharpest month-over-month avg_score drops (and which issue-category co-spiked) ===")
for app_name, g in monthly.groupby("app_name"):
    g = g.sort_values("month").reset_index(drop=True)
    g["score_delta"] = g["avg_score"].diff()
    worst = g.sort_values("score_delta").head(2)
    for _, r in worst.iterrows():
        if pd.isna(r["score_delta"]):
            continue
        # which issue category has the highest pct in this month
        issue_cols = {k: r[f"pct_{k}"] for k in ISSUE_KEYWORDS}
        top_issue = max(issue_cols, key=issue_cols.get)
        print(f"{app_name:10s} {r['month']}  avg_score={r['avg_score']:.2f}  (Δ{r['score_delta']:+.2f})  "
              f"n={r['review_count']:.0f}  version~{r['dominant_version']}  top_issue={top_issue} ({issue_cols[top_issue]:.0%} of reviews)")

print("\n=== Overall correlation: avg_score vs any_issue_flagged rate (per app-month) ===")
monthly["pct_any_issue"] = monthly[[f"pct_{k}" for k in ISSUE_KEYWORDS]].sum(axis=1)
for app_name, g in monthly.groupby("app_name"):
    if len(g) >= 4:
        corr = g["avg_score"].corr(g["pct_any_issue"])
        print(f"{app_name:10s} corr(avg_score, issue_mention_rate) = {corr:+.2f}   (n_months={len(g)})")
