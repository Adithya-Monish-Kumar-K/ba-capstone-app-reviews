import time
import pandas as pd
from google_play_scraper import reviews, Sort, app as app_info

APPS = {
    "in.swiggy.android": "Swiggy",
    "com.application.zomato": "Zomato",
    "com.myntra.android": "Myntra",
    "net.one97.paytm": "Paytm",
    "com.phonepe.app": "PhonePe",
}

TARGET_PER_APP = 3000
BATCH = 200

all_rows = []
app_meta_rows = []

for pkg, name in APPS.items():
    print(f"=== {name} ({pkg}) ===")
    try:
        meta = app_info(pkg, lang="en", country="in")
        app_meta_rows.append({
            "app_id": pkg, "app_name": meta["title"], "category": meta["genre"],
            "current_score": meta["score"], "installs": meta["installs"],
            "total_ratings": meta["ratings"],
        })
    except Exception as e:
        print("  meta FAIL", e)
        continue

    collected = []
    token = None
    tries = 0
    while len(collected) < TARGET_PER_APP and tries < 20:
        try:
            batch, token = reviews(
                pkg, lang="en", country="in", sort=Sort.MOST_RELEVANT,
                count=BATCH, continuation_token=token
            )
        except Exception as e:
            print("  page FAIL", e)
            break
        if not batch:
            break
        collected.extend(batch)
        tries += 1
        print(f"  page {tries}: +{len(batch)} (total {len(collected)})")
        if token is None:
            break
        time.sleep(0.3)

    for rv in collected:
        all_rows.append({
            "app_id": pkg,
            "app_name": name,
            "review_id": rv["reviewId"],
            "score": rv["score"],
            "content": rv["content"],
            "thumbs_up": rv["thumbsUpCount"],
            "app_version": rv["reviewCreatedVersion"],
            "review_date": rv["at"],
        })
    print(f"  TOTAL collected for {name}: {len(collected)}")

df = pd.DataFrame(all_rows)
out = "/home/Adithya/Desktop/Sem 7/BA/BAProject/topic1_app_review_dataset/app_reviews_raw.csv"
df.to_csv(out, index=False)
print("\nSaved", len(df), "rows to", out)

meta_df = pd.DataFrame(app_meta_rows)
meta_out = "/home/Adithya/Desktop/Sem 7/BA/BAProject/topic1_app_review_dataset/app_metadata.csv"
meta_df.to_csv(meta_out, index=False)
print("Saved", len(meta_df), "rows to", meta_out)

print("\nDate range per app:")
df["review_date"] = pd.to_datetime(df["review_date"])
print(df.groupby("app_name")["review_date"].agg(["min", "max", "count"]))
