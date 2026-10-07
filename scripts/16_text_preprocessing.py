"""
Stage 1 of Review 2, Method 1 (Text Mining): data preparation.

Prepares the review text for topic modelling:
  * complaint corpus  = 1-2 star reviews long enough to carry a topic
  * praise corpus     = 4-5 star reviews, as a contrast (what users praise)
Each review is lowercased, contractions are expanded ("didn't" -> "did not"), emojis, digits and punctuation
are removed, stop words are dropped except negations ("not", "no", "never"), and every word is lemmatised
("delivered" -> "deliver", "orders" -> "order"). App and brand names are removed so that topics describe
problems, not apps.

Input:  data/tagged/<app>.csv.gz
Output: data/textmining/
          corpus.csv.gz                one row per kept review: ids, app, domain, rating, date, sentiment,
                                       group (complaint / praise), clean_text, n_tokens
          dtm/<group>_<domain>.npz     document-term matrix (word and two-word-phrase counts) per domain
          dtm/<group>_<domain>_vocab.json, dtm/<group>_<domain>_rows.csv.gz   its vocabulary and review_id order
        data/charts/textmining/tm_01_top_terms.png
        data/textmining_summary.json

Load a matrix:  X = scipy.sparse.load_npz(path); vocab = json.load(open(vocab_path)); rows = pd.read_csv(rows_path)
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apps import APP_NAMES, DOMAIN_COLORS, DOMAINS  # noqa: E402
from data_io import read_stage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "textmining"
DTM_DIR = OUT_DIR / "dtm"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "textmining"
SUMMARY_PATH = REPO_ROOT / "data" / "textmining_summary.json"
DOMAIN_SLUG = {"Food & Grocery": "food_grocery", "Shopping": "shopping", "Payments": "payments"}

MIN_TOKENS = 3            # a review needs at least 3 content words (after cleaning) to carry a topic
MIN_DF, MAX_DF = 10, 0.5  # a term must appear in >= 10 reviews and in at most half of them
NGRAMS = (1, 2)           # words and two-word phrases ("not deliver", "customer care")

NEGATIONS = {"not", "no", "never", "nor", "none", "nothing", "nobody", "cannot", "without"}
CONTRACTIONS = [
    (r"\bcan'?t\b", "can not"), (r"\bwon'?t\b", "will not"), (r"\bshan'?t\b", "shall not"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|should|would|could|must|need)n'?t\b", r"\1 not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"),
    (r"'m\b", " am"), (r"'s\b", ""),
]
# App and brand names (and common spellings) say which app a review is about, not what went wrong.
BRANDS = {"swiggy", "zomato", "blinkit", "grofers", "domino", "dominos", "myntra", "flipkart", "amazon", "meesho",
          "paytm", "phonepe", "phonpe", "gpay", "googlepay", "instamart", "zepto", "amazonpay"}
# Two-word brand names are removed as phrases, so that "pay" and "google" stay usable on their own.
BRAND_PHRASES = r"\b(google\s*pay|phone\s*pe|amazon\s*pay|g\s*pay)\b"
EXTRA_STOP = {"app", "application", "apps", "pls", "plz", "please", "also", "even", "ok", "okay", "u", "ur",
              "dont", "doesnt", "didnt", "isnt", "wasnt", "cant", "wont", "im", "ive", "thats", "hai", "h", "soo"}

# Hinglish: negations map to "not" (their meaning is kept); common function words are dropped.
HINGLISH_NOT = {"nahi", "nahin", "nhi", "nai", "nahee", "nahiin", "mat"}
HINGLISH_STOP = {"hai", "hain", "ho", "hua", "hui", "hue", "raha", "rahi", "rahe", "ka", "ki", "ke", "ko", "se", "me",
                 "mein", "main", "mai", "mera", "meri", "mere", "bhi", "aur", "ye", "yeh", "wo", "woh", "to", "hi", "tha",
                 "thi", "the", "kar", "karo", "karna", "kiya", "kya", "kyon", "kyu", "kyun", "par", "per", "ek", "koi",
                 "sab", "bahut", "bhai", "aap", "apna", "apne", "tum", "hum", "ham", "kuch", "ab", "jo", "liye", "gaya",
                 "gayi", "ja", "jata", "jati", "rha", "rhi", "hota", "hoti", "kr", "ke", "bhi", "ji", "sir", "mam"}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "legend.frameon": False,
})


def nltk_resources():
    for res, path in [("wordnet", "corpora/wordnet"), ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop = (set(stopwords.words("english")) | EXTRA_STOP | BRANDS | HINGLISH_STOP) - NEGATIONS
    return stop, WordNetLemmatizer()


STOP, LEMMATIZER = nltk_resources()
_LEMMA_CACHE = {}


def lemma(word):
    """Verb lemma, then noun lemma: 'delivered' -> 'deliver', 'orders' -> 'order', 'refunded' -> 'refund'."""
    if word not in _LEMMA_CACHE:
        _LEMMA_CACHE[word] = LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(word, "v"), "n")
    return _LEMMA_CACHE[word]


def clean(text):
    t = str(text).lower().replace("’", "'")
    t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", t)
    t = re.sub(BRAND_PHRASES, " ", t)
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\b(dont|doesnt|didnt|isnt|wasnt|cant|wont|havent|hasnt|shouldnt|wouldnt|couldnt)\b",
               lambda m: {"cant": "can", "wont": "will"}.get(m.group(1), m.group(1)[:-2]) + " not", t)
    t = re.sub(r"[^a-z\s]", " ", t)                 # emojis, digits and punctuation
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)            # "sooooo" -> "soo"
    words = ["not" if w in HINGLISH_NOT else w for w in t.split()]
    words = [w for w in words if len(w) > 1 and w not in STOP]
    words = [lemma(w) for w in words]
    return [w for w in words if len(w) > 1 and w not in STOP]


def build_corpus():
    cols = ["review_id", "app_name", "domain", "score", "content", "review_date", "review_length", "sentiment_compound"]
    df = read_stage("tagged", columns=cols, parse_dates=["review_date"])
    df["group"] = np.where(df["score"] <= 2, "complaint", np.where(df["score"] >= 4, "praise", "neutral"))
    df = df[df["group"] != "neutral"].copy()
    print(f"  cleaning {len(df):,} reviews (1-2 and 4-5 stars)...")
    tokens = df["content"].map(clean)
    df["n_tokens"] = tokens.map(len)
    df["clean_text"] = tokens.map(" ".join)
    return df


def build_dtms(kept):
    info = {}
    DTM_DIR.mkdir(parents=True, exist_ok=True)
    for group in ["complaint", "praise"]:
        for dom in DOMAINS:
            d = kept[(kept["group"] == group) & (kept["domain"] == dom)]
            vec = CountVectorizer(ngram_range=NGRAMS, min_df=MIN_DF, max_df=MAX_DF, token_pattern=r"\S+", dtype=np.int32)
            X = vec.fit_transform(d["clean_text"])
            name = f"{group}_{DOMAIN_SLUG[dom]}"
            sparse.save_npz(DTM_DIR / f"{name}.npz", X)
            vocab = vec.get_feature_names_out().tolist()
            (DTM_DIR / f"{name}_vocab.json").write_text(json.dumps(vocab))
            d[["review_id", "app_name"]].to_csv(DTM_DIR / f"{name}_rows.csv.gz", index=False, compression="gzip")
            df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
            top = np.argsort(-df_counts)[:20]
            info[name] = {"group": group, "domain": dom, "reviews": int(X.shape[0]), "terms": int(X.shape[1]),
                          "bigrams": int(sum(" " in v for v in vocab)),
                          "nonzero_per_review": round(float(X.getnnz() / X.shape[0]), 2),
                          "top_terms_by_reviews": [[vocab[i], int(df_counts[i])] for i in top]}
            print(f"  {name}: {X.shape[0]:,} reviews x {X.shape[1]:,} terms")
    return info


def chart_top_terms(dtm_info):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, group in enumerate(["complaint", "praise"]):
        for col, dom in enumerate(DOMAINS):
            ax = axes[row, col]
            info = dtm_info[f"{group}_{DOMAIN_SLUG[dom]}"]
            terms = info["top_terms_by_reviews"][:12][::-1]
            share = [c / info["reviews"] * 100 for _, c in terms]
            ax.barh([t for t, _ in terms], share, color=DOMAIN_COLORS[dom] if group == "complaint" else "#9ec5f4", height=0.7)
            for y, v in enumerate(share):
                ax.text(v, y, f" {v:.0f}%", va="center", fontsize=8.5, color=INK2)
            ax.set_title(f"{dom}: {'1–2★ complaints' if group == 'complaint' else '4–5★ praise'} "
                         f"({info['reviews']:,})", loc="left", fontsize=10.5, color=INK2)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, max(share) * 1.2)
    axes[1, 0].set_xlabel("% of reviews in the corpus containing the term")
    title = "Most frequent terms after cleaning: complaints name the service, praise is generic"
    sub = ("Share of reviews containing each term after lowercasing, removing stop words (negations kept), app names, "
           "emojis and digits, and lemmatising. Complaint corpora: 1–2★ reviews with at least 3 content words.")
    h = fig.get_figheight()
    lines = textwrap.wrap(sub, int(fig.get_figwidth() * 13.5))
    fig.tight_layout(rect=[0, 0, 1, 1 - (0.95 + 0.17 * (len(lines) - 1)) / h])
    fig.text(0.012, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.46 / h, "\n".join(lines), ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.4)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "tm_01_top_terms.png", dpi=150)
    plt.close(fig)
    print("  saved data/charts/textmining/tm_01_top_terms.png")
    return {"file": "data/charts/textmining/tm_01_top_terms.png", "title": title, "subtitle": sub}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and cleaning reviews...")
    df = build_corpus()
    df["kept"] = df["n_tokens"] >= MIN_TOKENS
    kept = df[df["kept"]].copy()
    kept["app_name"] = pd.Categorical(kept["app_name"], APP_NAMES, ordered=True)
    kept = kept.sort_values(["group", "domain", "app_name", "review_date"]).reset_index(drop=True)
    out_cols = ["review_id", "app_name", "domain", "score", "review_date", "sentiment_compound", "group",
                "n_tokens", "clean_text"]
    kept[out_cols].to_csv(OUT_DIR / "corpus.csv.gz", index=False, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
    print(f"  kept {len(kept):,} reviews -> data/textmining/corpus.csv.gz")

    print("Document-term matrices per domain...")
    dtm_info = build_dtms(kept)
    chart = chart_top_terms(dtm_info)

    sel = df.groupby("group").agg(reviews=("kept", "size"), kept=("kept", "sum"))
    sel["dropped_too_short"] = sel["reviews"] - sel["kept"]
    sel["kept_pct"] = (sel["kept"] / sel["reviews"] * 100).round(1)
    by_app = kept.groupby(["group", "app_name"], observed=True).size().unstack(0).fillna(0).astype(int)
    examples = (df[df["group"] == "complaint"].sample(8, random_state=7)[["app_name", "content", "clean_text", "kept"]]
                .to_dict(orient="records"))
    summary = {
        "input_reviews_1_2_and_4_5_star": int(len(df)),
        "rules": {"min_tokens": MIN_TOKENS, "min_df": MIN_DF, "max_df": MAX_DF, "ngram_range": list(NGRAMS),
                  "negations_kept": sorted(NEGATIONS), "brand_words_removed": sorted(BRANDS),
                  "brand_phrases_removed": ["google pay", "phone pe", "amazon pay", "g pay"],
                  "hinglish_negations_to_not": sorted(HINGLISH_NOT), "hinglish_stop_words": len(HINGLISH_STOP),
                  "lemmatiser": "WordNet (verb, then noun)"},
        "selection": {g: {k: (int(v) if k != "kept_pct" else float(v)) for k, v in r.items()} for g, r in sel.iterrows()},
        "median_tokens_kept": kept.groupby("group")["n_tokens"].median().to_dict(),
        "kept_by_app": {g: by_app[g].to_dict() for g in by_app.columns},
        "document_term_matrices": dtm_info,
        "cleaning_examples": examples,
        "charts": [chart],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    print(f"Saved summary to {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print(sel.to_string())


if __name__ == "__main__":
    main()
