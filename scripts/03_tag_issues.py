import json
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from apps import APPS
from data_io import DATA_DIR, read_app, read_stage, write_app

SUMMARY_PATH = DATA_DIR / "issue_tagging_summary.json"

# =============================================================================
# DOMAIN-SPECIFIC ISSUE TAXONOMY
# Covers 11 apps across Food & Grocery (Swiggy, Zomato, Blinkit, Domino's),
# Shopping (Myntra, Flipkart, Amazon, Meesho) and Payments (Paytm, PhonePe, Google Pay).
# =============================================================================
TAXONOMY = {
    "crash_bugs_stability": {
        "description": "App crashes, freezes, lagging, black screen, glitches, server errors, loading failures",
        "patterns": [
            r"\bcrash(es|ed|ing)?\b",
            r"\bfreez(e|es|ed|ing)?\b",
            r"\bfrozen\b",
            r"\bhang(s|ed|ing)?\b",
            r"\blag(s|ged|ging|gy)?\b",
            r"\b(black|blank|white)\s+screen\b",
            r"\bforce\s+close(d)?\b",
            r"\bkeeps\s+(stopping|crashing|closing|freezing)\b",
            r"\bnot\s+responding\b",
            r"\bglitch(es|y)?\b",
            r"\bbug(s|gy)?\b",
            r"\b(not|unable\s+to|cannot|can\'?t|won\'?t)\s+open\b",
            r"\b(app|application)\s+(closes|closed|stopped|stopped\s+working)\b",
            r"\bloading\s+(issue|problem|slow|error|stuck|takes\s+forever)\b",
            r"\bstuck\s+(on|at|while|loading)\b",
            r"\bserver\s+(down|error|busy|unavailable|issue|timeout)\b",
            r"\b(usb\s+debugging|developer\s+option|security\s+alert|device\s+environment)\b"
        ]
    },
    "payment_refund": {
        "description": "Payment failures, money deducted without order, refund delays/denials, UPI and wallet errors",
        "patterns": [
            r"\brefund(s|ed|ing)?\b",
            r"\b(money|amount|funds|cash)\s+(deducted|debited|cut|stuck|lost|gone|missing|not\s+refunded|not\s+credited)\b",
            r"\b(payment|transaction)\s+(failed|failure|declined|pending|error|issue|problem|stuck|cancelled)\b",
            r"\b(failed|declined|unsuccessful)\s+(payment|transaction)\b",
            r"\bdouble\s+(payment|charge|deduction)\b",
            r"\bcharged?\s+(twice|double|extra)\b",
            r"\bupi\s+(fail|failure|issue|error|pending|declined|down|timeout|not\s+working|stops?\s+working)\b",
            r"\b(wallet|bank)\s+(balance|transfer|issue|pending)\b",
            r"\bchargeback\b",
            r"\bpayment\s+gateway\b",
            r"\bunauthorized\s+(transaction|charge|payment)\b",
            r"\bpayment\s+(showed|shows|status\s+is)?\s*(as\s+)?pending\b",
            r"\bmoney\s+not\s+(credited|received|returned|refunded)\b"
        ]
    },
    "delivery_delay": {
        "description": "Late delivery, excessive waiting time, slow delivery partners/riders, GPS tracking inaccuracies",
        "patterns": [
            r"\blate\s+delivery\b",
            r"\bdelay(ed|s|ing)?\b",
            r"\b(slow|poor|pathetic|worst)\s+delivery\b",
            r"\bdelivery\s+(delay|delayed|slow|late|time|taking\s+too\s+long|takes\s+long|timing)\b",
            r"\b(rider|delivery\s+(boy|partner|guy|person|agent|driver))\b",
            r"\b(live\s+)?tracking\s+(not\s+working|wrong|fake|stuck|issue|inaccurate)\b",
            r"\b(waiting|waited)\s+for\s+(order|delivery|hours|food)\b",
            r"\bhours\s+to\s+deliver\b",
            r"\bnot\s+delivered\s+on\s+time\b",
            r"\btime\s+(taken|taking)\s+too\s+long\b",
            r"\bestimated\s+(delivery\s+)?time\b",
            r"\btook\s+(more\s+than\s+)?\d+\s*(mins?|minutes?|hours?)\s+to\s+deliver\b"
        ]
    },
    "order_quality_fulfillment": {
        "description": "Wrong items, missing items, damaged packaging, stale/cold food, fake products, false delivered status",
        "patterns": [
            r"\bwrong\s+(item|items|food|order|product|products|size|colour|color|dish)\b",
            r"\bmissing\s+(item|items|food|order|product|products|part|quantity)\b",
            r"\b(item|items|product|food)\s+missing\b",
            r"\bdamaged\s+(item|items|product|products|package|parcel|box|food)\b",
            r"\b(poor|bad|pathetic|terrible|horrible|cheap)\s+quality\b",
            r"\b(stale|cold|spoiled|smelly|unhygienic|rotten|tasteless|undercooked)\s+food\b",
            r"\bspill(ed|age)?\b",
            r"\bexpired\s+(item|product|food)?\b",
            r"\bfake\s+(product|brand|item)\b",
            r"\bdefective\b",
            r"\bdelivered\s+without\s+(receiving|delivering|order)\b",
            r"\bmarked\s+(as\s+)?delivered\s+(without|but|never)\b",
            r"\bfake\s+delivery\b",
            r"\b(did\s+not|never)\s+receive(d)?\s+(the\s+)?(order|item|food|product)\b",
            r"\bhalf\s+(order|items)\b"
        ]
    },
    "cancellation_return": {
        "description": "Order cancellations, cancellation charges, return request rejections, failed reverse pickups, exchange problems",
        "patterns": [
            r"\bcancel(led|ling|lation)?\b",
            r"\bcancellation\s+fee(s)?\b",
            r"\bunable\s+to\s+cancel\b",
            r"\breturn(s|ed|ing)?\b",
            r"\breturn\s+(rejected|declined|request|policy|pickup|issue|problem|failed)\b",
            r"\bexchange(d|s|ing)?\b",
            r"\breplacement\b",
            r"\breplace(d|ing)?\b",
            r"\b(pickup|pick\s+up)\s+(not\s+done|delayed|failed|cancelled|pending)\b",
            r"\breverse\s+pickup\b"
        ]
    },
    "customer_support": {
        "description": "Unresponsive customer care, useless automated bots, closed tickets without resolution, rude executives",
        "patterns": [
            r"\bcustomer\s+(care|support|service)s?\b",
            r"\bsupport\s+(team|executive|agent|help|chat|desk)s?\b",
            r"\b(chat)?bot(s)?\b",
            r"\bautomated\s+(reply|response|message|answers?)\b",
            r"\bno\s+(response|reply|help|solution|support|resolution)\b",
            r"\b(worst|poor|bad|pathetic|terrible|useless|horrible|hopeless|third\s+class)\s+(customer\s+)?(care|service|support)s?\b",
            r"\b(raise|raised|open|opened)\s+(a\s+)?(ticket|complaint)\b",
            r"\bhelpline\b",
            r"\bclosed\s+(the\s+)?(ticket|complaint|case)\b",
            r"\bunhelpful\s+(support|executive|agent)?\b",
            r"\brude\s+(support|executive|agent|behaviour|behavior|person)\b",
            r"\bno\s+(one|body)\s+(helps|responds|replies|picks\s+up|answers)\b",
            r"\bcontact\s+(support|customer\s+care)\b"
        ]
    },
    "account_login_otp": {
        "description": "OTP not arriving, login failures, blocked/suspended accounts, KYC verification errors, password reset",
        "patterns": [
            r"\botp\b",
            r"\b(verification|security)\s+code\b",
            r"\b(can\'?t|cannot|unable\s+to|not\s+able\s+to)\s+(log\s*in|sign\s*in|login|signin)\b",
            r"\blogin\s+(issue|problem|failed|failure|error|attempt(s)?)\b",
            r"\b(account|number)\s+(blocked|locked|suspended|banned|disabled|frozen)\b",
            r"\bkyc\b",
            r"\bpassword\s+(reset|error|forgot)\b",
            r"\blogged\s+out\b",
            r"\bdevice\s+(verification|registered|limit|binding)\b",
            r"\bsim\s+(binding|verification|card)\b"
        ]
    },
    "pricing_charges_fraud": {
        "description": "Hidden charges, platform/delivery fees, surge pricing, overcharging, coupons not working, scam/fraud allegations",
        "patterns": [
            r"\b(hidden|extra|excessive|additional|unnecessary)\s+(charges?|fees?|cost)\b",
            r"\b(platform|delivery|handling|packaging|convenience|surge)\s+fees?\b",
            r"\b(overpriced|overcharging|overcharged)\b",
            r"\b(high|expensive)\s+(rates?|price|pricing|charge)\b",
            r"\b(scam|cheating|fraud|looting|loot|cheat|scammer|thieves|theft)\b",
            r"\bcoupon(s)?\s+(not\s+working|invalid|fake|issue|expired)\b",
            r"\b(discount|offer)\s+(fake|not\s+applied|cheated)\b",
            r"\bcashback\s+(not\s+received|fake|fraud|pending)\b",
            r"\bcharging\s+(extra|more)\b"
        ]
    },
    "ui_ux_update": {
        "description": "Post-update regressions, poor interface/navigation, search/filter broken, removed features, intrusive ads",
        "patterns": [
            r"\b(new|latest|recent)\s+update\b",
            r"\bafter\s+(the\s+|this\s+|new\s+)?update\b",
            r"\b(worst|terrible|horrible|bad)\s+update\b",
            r"\bupdate\s+(ruined|broke|destroyed|spoiled)\b",
            r"\b(user\s+interface|ui|ux)\b",
            r"\b(interface|navigation|layout|design)\s+(is\s+)?(bad|confusing|horrible|terrible|poor|cluttered|worst)\b",
            r"\bsearch\s+(bar|option|not\s+working|broken|feature)\b",
            r"\bfilter(s)?\s+(not\s+working|removed|issue)\b",
            r"\bfeature(s)?\s+(removed|missing|gone)\b",
            r"\b(annoying|too\s+many|excessive)\s+ads?\b",
            r"\bkeeps?\s+asking\s+(me\s+)?to\s+update\b"
        ]
    }
}

# Compile regular expressions
COMPILED_PATTERNS = {
    category: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
    for category, data in TAXONOMY.items()
}


def tag_review_text(text: str) -> dict:
    """
    Applies taxonomy rules to a review text string.
    Returns a dictionary of category hit counts and boolean flags.
    """
    cleaned_text = str(text) if pd.notna(text) else ""
    tag_counts = {}
    tag_flags = {}

    for category, regex_list in COMPILED_PATTERNS.items():
        count = sum(1 for reg in regex_list if reg.search(cleaned_text))
        tag_counts[category] = count
        tag_flags[f"issue_{category}"] = 1 if count > 0 else 0

    detected_issues = [cat for cat, cnt in tag_counts.items() if cnt > 0]
    issue_count = len(detected_issues)
    has_issue = 1 if issue_count > 0 else 0

    if issue_count > 0:
        # Primary issue is the category with the most matching pattern hits
        primary_issue = max(tag_counts, key=tag_counts.get)
        all_issues_str = ";".join(detected_issues)
    else:
        primary_issue = "none"
        all_issues_str = "none"

    return {
        **tag_flags,
        "issue_count": issue_count,
        "has_issue": has_issue,
        "primary_issue": primary_issue,
        "all_issues": all_issues_str
    }


def tag_app(slug):
    df = read_app("clean", slug)
    tag_df = pd.DataFrame(df["content"].apply(tag_review_text).tolist())
    write_app(pd.concat([df, tag_df], axis=1), "tagged", slug)
    return slug, len(df)


def main():
    print("Applying keyword taxonomy rules to data/clean/*.csv.gz (one process per app)...")
    with ProcessPoolExecutor(max_workers=11) as pool:
        for slug, n in pool.map(tag_app, APPS):
            print(f"  {APPS[slug]['name']:10s} {n:>9,} reviews tagged")

    issue_cols = [f"issue_{c}" for c in TAXONOMY]
    tagged_df = read_stage("tagged", columns=["app_name", "issue_count", "has_issue"] + issue_cols)
    n_total = len(tagged_df)

    total_tagged = int((tagged_df["has_issue"] == 1).sum())
    coverage_pct = round(total_tagged / n_total * 100, 2)

    category_summary = {}
    for cat in TAXONOMY:
        col = f"issue_{cat}"
        count = int(tagged_df[col].sum())
        category_summary[cat] = {
            "count": count,
            "pct_of_total": round(count / n_total * 100, 2),
            "description": TAXONOMY[cat]["description"]
        }

    app_breakdown = {}
    for app_name, group in tagged_df.groupby("app_name"):
        app_total = len(group)
        app_tagged = int((group["has_issue"] == 1).sum())
        app_cat_counts = {cat: int(group[f"issue_{cat}"].sum()) for cat in TAXONOMY}
        app_breakdown[app_name] = {
            "total_reviews": app_total,
            "tagged_reviews": app_tagged,
            "coverage_pct": round(app_tagged / app_total * 100, 2),
            "top_categories": sorted(app_cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    summary = {
        "input_rows": n_total,
        "tagged_rows": total_tagged,
        "overall_coverage_pct": coverage_pct,
        "avg_issues_per_review": round(float(tagged_df["issue_count"].mean()), 2),
        "category_summary": category_summary,
        "app_breakdown": app_breakdown
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TAGGING COMPLETE SUMMARY")
    print("=" * 60)
    print(f"Total reviews: {n_total:,}")
    print(f"Tagged with >= 1 issue: {total_tagged:,} ({coverage_pct}%)")
    print(f"Average issues per review: {summary['avg_issues_per_review']}")
    print("\nCategory Distribution:")
    for cat, info in category_summary.items():
        print(f"  {cat:28s}: {info['count']:9,d} ({info['pct_of_total']:5.1f}%)")

    print("\nApp Breakdown:")
    for app, info in app_breakdown.items():
        top_str = ", ".join(f"{c} ({cnt})" for c, cnt in info["top_categories"])
        print(f"  {app:10s}: {info['coverage_pct']:5.1f}% tagged | Top: {top_str}")

    print("\nSaved output dataset: data/tagged/*.csv.gz")
    print(f"Saved summary metrics: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
