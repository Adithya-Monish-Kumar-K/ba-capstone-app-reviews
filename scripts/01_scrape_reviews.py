"""
Stage 1: Play Store review collection (Person 1)

Collects EVERY English-language India review posted from START_DATE to END_DATE (inclusive) for each app,
using Sort.NEWEST and paginating backwards in time until a whole page is older than START_DATE.
A fixed date window (instead of a fixed review count) gives every app identical time coverage.

Output: data/raw/<app_slug>.csv.gz (one file per app, so each stays under GitHub's 100 MB limit)
        data/app_metadata.csv

Usage:  python scripts/01_scrape_reviews.py              # all apps
        python scripts/01_scrape_reviews.py flipkart amazon   # a subset (lets apps run in parallel)
Apps whose output file already exists are skipped, so an interrupted run can simply be restarted.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, app as app_info, reviews

from apps import APPS

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
META_DIR = REPO_ROOT / "data" / "raw" / "metadata"
START_DATE = datetime(2026, 4, 1)
END_DATE = datetime(2026, 9, 21)    # exclusive: the window ends on 20 Sep 2026
BATCH = 200
MAX_RETRIES = 6
EMPTY_PAGE_RETRIES = 5


def fetch_page(pkg, token):
    for attempt in range(MAX_RETRIES):
        try:
            return reviews(pkg, lang="en", country="in", sort=Sort.NEWEST,
                           count=BATCH, continuation_token=token)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e!r}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{pkg}: page failed after {MAX_RETRIES} retries")


def scrape_app(slug, info):
    out = RAW_DIR / f"{slug}.csv.gz"
    if out.exists():
        print(f"=== {info['name']}: already scraped, skipping", flush=True)
        return

    meta = app_info(info["package"], lang="en", country="in")
    pd.DataFrame([{
        "app_id": info["package"], "app_name": info["name"], "domain": info["domain"],
        "store_title": meta["title"], "category": meta["genre"], "current_score": meta["score"],
        "installs": meta["installs"], "total_ratings": meta["ratings"],
        "metadata_collected_at": datetime.now().isoformat(timespec="seconds"),
    }]).to_csv(META_DIR / f"{slug}.csv", index=False)

    rows, token, pages, t0 = [], None, 0, time.time()
    while True:
        # The endpoint occasionally returns an empty page mid-stream; retry the same token before giving up.
        for attempt in range(EMPTY_PAGE_RETRIES):
            batch, next_token = fetch_page(info["package"], token)
            if batch:
                break
            time.sleep(2 ** attempt)
        pages += 1
        if not batch:
            stop = "empty pages after retries"
            break
        token = next_token
        rows.extend(r for r in batch if START_DATE <= r["at"] < END_DATE)
        oldest = min(r["at"] for r in batch)
        if pages % 50 == 0:
            print(f"  {info['name']}: page {pages}, {len(rows):,} kept, reached {oldest:%Y-%m-%d}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        if oldest < START_DATE:
            stop = "reached START_DATE"
            break
        if token is None:
            stop = "no continuation token"
            break

    df = pd.DataFrame([{
        "app_id": info["package"],
        "app_name": info["name"],
        "domain": info["domain"],
        "review_id": r["reviewId"],
        "score": r["score"],
        "content": r["content"],
        "thumbs_up": r["thumbsUpCount"],
        "app_version": r["reviewCreatedVersion"],
        "review_date": r["at"],
    } for r in rows]).drop_duplicates(subset="review_id")

    tmp = out.with_suffix(".tmp.gz")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(out)
    print(f"=== {info['name']}: {len(df):,} reviews ({df['review_date'].min():%Y-%m-%d} to "
          f"{df['review_date'].max():%Y-%m-%d}), {pages} pages, {time.time() - t0:.0f}s, stop: {stop}", flush=True)


def write_metadata():
    files = sorted(META_DIR.glob("*.csv"))
    if files:
        pd.concat(pd.read_csv(f) for f in files).to_csv(REPO_ROOT / "data" / "app_metadata.csv", index=False)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    selected = sys.argv[1:] or list(APPS)
    for slug in selected:
        scrape_app(slug, APPS[slug])
    write_metadata()
