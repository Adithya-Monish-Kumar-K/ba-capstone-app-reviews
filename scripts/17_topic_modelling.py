"""
Stage 2 of Review 2, Method 1 (Text Mining): Topic Modelling & Distinctive Terms (#134).

Fits Non-negative Matrix Factorization (NMF) topic models on 1-2 star complaints per domain,
extracts top terms and exemplar reviews, computes distinctive words/phrases per app using
Monroe et al. (2008) log-odds with Dirichlet prior, quantifies sentiment per topic and per app,
models 4-5 star praise as a contrast, and saves deployable topic inference pipelines for the dashboard.

Input:  data/textmining/dtm/<group>_<domain>.npz, _vocab.json, _rows.csv.gz
        data/textmining/corpus.csv.gz
Output: models/textmining/topic_pipeline_<domain_slug>.pkl (deployable inference pipelines)
        data/textmining/
          topics.csv                   top terms and loadings for each topic
          topic_exemplars.csv          top exemplar reviews per topic
          distinctive_terms.csv        top distinctive words & phrases per app (log-odds)
          topic_sentiment.csv          sentiment metrics per topic and app
          review_topics.csv.gz         dominant topic assignments for all complaint reviews
        data/charts/textmining/
          tm_02_complaint_topics.png   top terms and loadings per complaint topic
          tm_03_distinctive_terms.png  distinctive terms per app (log-odds z-score)
          tm_04_praise_topics.png      topics of praise (4-5 star contrast)
          tm_05_topic_sentiment.png    mean sentiment & strongly negative share per topic
        data/topic_modelling_summary.json

Usage:
  python scripts/17_topic_modelling.py
  python scripts/17_topic_modelling.py --predict "Refund not received, customer care never replies" --domain "Payments"
"""
import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from apps import APP_COLORS, APP_DOMAIN, APP_NAMES, APPS, DOMAIN_COLORS, DOMAINS  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "textmining"
DTM_DIR = DATA_DIR / "dtm"
MODELS_DIR = REPO_ROOT / "models" / "textmining"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "textmining"
SUMMARY_PATH = REPO_ROOT / "data" / "topic_modelling_summary.json"
CORPUS_PATH = DATA_DIR / "corpus.csv.gz"

DOMAIN_SLUG = {"Food & Grocery": "food_grocery", "Shopping": "shopping", "Payments": "payments"}
SLUG_DOMAIN = {v: k for k, v in DOMAIN_SLUG.items()}
GZIP = {"method": "gzip", "mtime": 0}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "legend.frameon": False,
})

# Predefined domain topic counts and canonical labels based on domain coherence and taxonomy alignment
COMPLAINT_CONFIG = {
    "Food & Grocery": {
        "k": 6,
        "labels": [
            "Order Cancellation & Refund",
            "Customer Support & Bot Loop",
            "Excessive Delivery & Handling Charges",
            "Poor Food Quality & Stale Items",
            "Delivery Delays & Rider Tracking",
            "COD & Payment Option Failures",
        ],
    },
    "Shopping": {
        "k": 6,
        "labels": [
            "Return Rejection & Defective Products",
            "Customer Service & Escalation Failure",
            "Courier Delays & Delivery Agent Issues",
            "Poor Shopping Experience & Fabric Quality",
            "Extreme Dissatisfaction & Scam Allegations",
            "Order Cancellation & Rescheduling",
        ],
    },
    "Payments": {
        "k": 5,
        "labels": [
            "Failed Transfers & Debited Amounts",
            "App Crashes & Update Regressions",
            "Device Environment & Security Scan Errors",
            "Account Blocking & AutoPay Issues",
            "Customer Support Unresponsiveness",
        ],
    },
}

PRAISE_CONFIG = {
    "Food & Grocery": {
        "k": 4,
        "labels": [
            "Lightning Fast Delivery",
            "Helpful Service & Fresh Quality",
            "Tasty Food & Great Orders",
            "Top-Tier Overall Experience",
        ],
    },
    "Shopping": {
        "k": 4,
        "labels": [
            "Great Product & Prompt Service",
            "Trusted Online Shopping Platform",
            "High Fabric & Build Quality",
            "Pleasant Buying Experience",
        ],
    },
    "Payments": {
        "k": 4,
        "labels": [
            "Smooth & Reliable Experience",
            "Fast UPI & Digital Transactions",
            "Effortless & Intuitive UI",
            "Trusted Cashless Convenience",
        ],
    },
}


# =============================================================================
# Helper: Text cleaning for new reviews (identical to scripts/16_text_preprocessing.py)
# =============================================================================
def _get_cleaner():
    import nltk
    for res, p in [("wordnet", "corpora/wordnet"), ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(p)
        except LookupError:
            nltk.download(res, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    negations = {"not", "no", "never", "nor", "none", "nothing", "nobody", "cannot", "without"}
    brands = {"swiggy", "zomato", "blinkit", "grofers", "domino", "dominos", "myntra", "flipkart", "amazon", "meesho",
              "paytm", "phonepe", "phonpe", "gpay", "googlepay", "instamart", "zepto", "amazonpay"}
    brand_phrases = r"\b(google\s*pay|phone\s*pe|amazon\s*pay|g\s*pay)\b"
    extra_stop = {"app", "application", "apps", "pls", "plz", "please", "also", "even", "ok", "okay", "u", "ur",
                  "dont", "doesnt", "didnt", "isnt", "wasnt", "cant", "wont", "im", "ive", "thats", "hai", "h", "soo"}
    hinglish_not = {"nahi", "nahin", "nhi", "nai", "nahee", "nahiin", "mat"}
    hinglish_stop = {"hai", "hain", "ho", "hua", "hui", "hue", "raha", "rahi", "rahe", "ka", "ki", "ke", "ko", "se", "me",
                     "mein", "main", "mai", "mera", "meri", "mere", "bhi", "aur", "ye", "yeh", "wo", "woh", "to", "hi", "tha",
                     "thi", "the", "kar", "karo", "karna", "kiya", "kya", "kyon", "kyu", "kyun", "par", "per", "ek", "koi",
                     "sab", "bahut", "bhai", "aap", "apna", "apne", "tum", "hum", "ham", "kuch", "ab", "jo", "liye", "gaya",
                     "gayi", "ja", "jata", "jati", "rha", "rhi", "hota", "hoti", "kr", "ke", "bhi", "ji", "sir", "mam"}
    contractions = [
        (r"\bcan'?t\b", "can not"), (r"\bwon'?t\b", "will not"), (r"\bshan'?t\b", "shall not"),
        (r"\b(do|does|did|is|are|was|were|has|have|had|should|would|could|must|need)n'?t\b", r"\1 not"),
        (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"),
        (r"'m\b", " am"), (r"'s\b", ""),
    ]
    stop = (set(stopwords.words("english")) | extra_stop | brands | hinglish_stop) - negations
    lemmatizer = WordNetLemmatizer()
    lemma_cache = {}

    def _lemma(w):
        if w not in lemma_cache:
            lemma_cache[w] = lemmatizer.lemmatize(lemmatizer.lemmatize(w, "v"), "n")
        return lemma_cache[w]

    def _clean(text):
        t = str(text).lower().replace("’", "'")
        t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", t)
        t = re.sub(brand_phrases, " ", t)
        for pat, rep in contractions:
            t = re.sub(pat, rep, t)
        t = re.sub(r"\b(dont|doesnt|didnt|isnt|wasnt|cant|wont|havent|hasnt|shouldnt|wouldnt|couldnt)\b",
                   lambda m: {"cant": "can", "wont": "will"}.get(m.group(1), m.group(1)[:-2]) + " not", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        words = ["not" if w in hinglish_not else w for w in t.split()]
        words = [w for w in words if len(w) > 1 and w not in stop]
        words = [_lemma(w) for w in words]
        return " ".join([w for w in words if len(w) > 1 and w not in stop])

    return _clean


# =============================================================================
# Topic Pipeline Class for Dashboard & Inference
# =============================================================================
class TopicPipeline:
    """Deployable topic pipeline containing vectorizer, tfidf and nmf for a single domain."""

    def __init__(self, domain, labels, vocab, nmf, tfidf):
        self.domain = domain
        self.labels = labels
        self.vocab = vocab
        self.nmf = nmf
        self.tfidf = tfidf
        self.vectorizer = CountVectorizer(vocabulary=vocab, token_pattern=r"\S+", ngram_range=(1, 2))

    def predict(self, raw_text, cleaner_fn=None):
        if cleaner_fn is None:
            cleaner_fn = _get_cleaner()
        cleaned = cleaner_fn(raw_text)
        if not cleaned.strip():
            return {
                "clean_text": "",
                "dominant_topic_id": 0,
                "dominant_topic": "Unassigned / Too Short",
                "confidence": 0.0,
                "weights": {lbl: 0.0 for lbl in self.labels},
            }
        counts = self.vectorizer.transform([cleaned])
        tfidf_vec = self.tfidf.transform(counts)
        w = self.nmf.transform(tfidf_vec)[0]
        s = float(w.sum())
        if s > 0:
            probs = w / s
            best_idx = int(np.argmax(w))
            conf = float(probs[best_idx])
            best_label = self.labels[best_idx]
            topic_id = best_idx + 1
        else:
            probs = np.zeros(len(self.labels))
            best_label = "Unassigned / General"
            topic_id = 0
            conf = 0.0
        return {
            "clean_text": cleaned,
            "dominant_topic_id": topic_id,
            "dominant_topic": best_label,
            "confidence": round(conf, 4),
            "weights": {self.labels[i]: round(float(probs[i]), 4) for i in range(len(self.labels))},
        }


# =============================================================================
# Core Pipeline: Fit Topic Models (NMF)
# =============================================================================
def fit_topic_models(group="complaint"):
    """Fits NMF on complaint or praise matrices per domain and returns models and topic metrics."""
    config = COMPLAINT_CONFIG if group == "complaint" else PRAISE_CONFIG
    results = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for domain, cfg in config.items():
        slug = DOMAIN_SLUG[domain]
        name = f"{group}_{slug}"
        print(f"Fitting NMF for {group} {domain} (K={cfg['k']})...")
        X = sparse.load_npz(DTM_DIR / f"{name}.npz")
        vocab = json.load(open(DTM_DIR / f"{name}_vocab.json", encoding="utf-8"))
        rows = pd.read_csv(DTM_DIR / f"{name}_rows.csv.gz")

        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X)

        nmf = NMF(n_components=cfg["k"], random_state=42, init="nndsvda", max_iter=250)
        W = nmf.fit_transform(X_tfidf)
        H = nmf.components_

        # Extract top terms per topic
        topics_info = []
        for t_idx in range(cfg["k"]):
            top_word_indices = H[t_idx].argsort()[:-16:-1]
            top_terms = [vocab[idx] for idx in top_word_indices]
            top_weights = [round(float(H[t_idx, idx]), 4) for idx in top_word_indices]
            topics_info.append({
                "topic_id": t_idx + 1,
                "label": cfg["labels"][t_idx],
                "top_terms": top_terms,
                "top_weights": top_weights,
            })

        # Build deployable pipeline
        pipeline = TopicPipeline(domain, cfg["labels"], vocab, nmf, tfidf)
        model_pkl = MODELS_DIR / f"topic_pipeline_{group}_{slug}.pkl"
        joblib.dump(pipeline, model_pkl)
        print(f"  saved deployable model -> {model_pkl.relative_to(REPO_ROOT)}")

        results[domain] = {
            "pipeline": pipeline,
            "W": W,
            "H": H,
            "vocab": vocab,
            "rows": rows,
            "topics": topics_info,
            "k": cfg["k"],
            "labels": cfg["labels"],
        }
    return results


# =============================================================================
# Distinctive Terms per App (Monroe et al. 2008 Log-Odds with Dirichlet Prior)
# =============================================================================
def compute_distinctive_terms():
    """Computes Monroe et al. (2008) standardized log-odds ratio with informative background Dirichlet prior."""
    print("Computing distinctive words and phrases per app (Monroe et al. 2008 log-odds)...")
    records = []

    for domain in DOMAINS:
        slug = DOMAIN_SLUG[domain]
        X = sparse.load_npz(DTM_DIR / f"complaint_{slug}.npz")
        vocab = np.array(json.load(open(DTM_DIR / f"complaint_{slug}_vocab.json", encoding="utf-8")))
        rows = pd.read_csv(DTM_DIR / f"complaint_{slug}_rows.csv.gz")

        domain_apps = [a for a in APP_NAMES if APP_DOMAIN[a] == domain]
        app_counts = {a: np.asarray(X[(rows["app_name"] == a).values].sum(axis=0)).ravel() for a in domain_apps}

        total_bg = np.asarray(X.sum(axis=0)).ravel()
        bg_sum = total_bg.sum()
        alpha_0 = 1000.0
        alpha = alpha_0 * (total_bg + 1.0) / (bg_sum + len(vocab))

        for app in domain_apps:
            y_i = app_counts[app]
            n_i = y_i.sum()
            y_j = total_bg - y_i
            n_j = bg_sum - n_i

            log_odds_i = np.log((y_i + alpha) / (n_i + alpha_0 - y_i - alpha))
            log_odds_j = np.log((y_j + alpha) / (n_j + alpha_0 - y_j - alpha))
            delta = log_odds_i - log_odds_j
            variance = (1.0 / (y_i + alpha)) + (1.0 / (y_j + alpha))
            z_scores = delta / np.sqrt(variance)

            top_indices = z_scores.argsort()[:-16:-1]
            for idx in top_indices:
                records.append({
                    "domain": domain,
                    "app_name": app,
                    "term": str(vocab[idx]),
                    "is_bigram": bool(" " in str(vocab[idx])),
                    "z_score": round(float(z_scores[idx]), 3),
                    "app_count": int(y_i[idx]),
                    "domain_count": int(total_bg[idx]),
                })
    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "distinctive_terms.csv", index=False)
    print(f"  saved {len(df)} distinctive terms -> data/textmining/distinctive_terms.csv")
    return df


# =============================================================================
# Sentiment by Topic and Exemplar Reviews
# =============================================================================
def evaluate_complaint_topics(complaint_models):
    """Assigns dominant topics to reviews, links with corpus sentiment, and extracts exemplar reviews."""
    print("Assigning dominant topics and computing topic sentiment...")
    corpus = pd.read_csv(CORPUS_PATH)
    corpus_lookup = corpus.set_index("review_id")

    assigned_frames = []
    exemplars = []
    topics_rows = []

    for domain, mod in complaint_models.items():
        W = mod["W"]
        rows = mod["rows"].copy()
        labels = mod["labels"]

        # Dominant topic assignment
        row_sums = W.sum(axis=1, keepdims=True)
        norm_W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)
        dominant_idx = np.argmax(W, axis=1)
        max_weight = np.max(norm_W, axis=1)

        rows["domain"] = domain
        rows["group"] = "complaint"
        rows["dominant_topic_id"] = dominant_idx + 1
        rows["dominant_topic_label"] = [labels[i] for i in dominant_idx]
        rows["topic_confidence"] = np.round(max_weight, 4)

        # Merge sentiment and content from corpus
        matched_sentiment = corpus_lookup.loc[rows["review_id"]]["sentiment_compound"].values
        matched_score = corpus_lookup.loc[rows["review_id"]]["score"].values
        rows["sentiment_compound"] = matched_sentiment
        rows["score"] = matched_score
        assigned_frames.append(rows)

        # Extract top exemplars for each topic
        for t_idx in range(mod["k"]):
            top_exemplar_indices = W[:, t_idx].argsort()[:-6:-1]
            for rank, r_idx in enumerate(top_exemplar_indices, 1):
                rid = rows.iloc[r_idx]["review_id"]
                rev = corpus_lookup.loc[rid]
                exemplars.append({
                    "domain": domain,
                    "topic_id": t_idx + 1,
                    "topic_label": labels[t_idx],
                    "rank": rank,
                    "loading": round(float(W[r_idx, t_idx]), 4),
                    "app_name": rev["app_name"],
                    "score": int(rev["score"]),
                    "sentiment_compound": round(float(rev["sentiment_compound"]), 4),
                    "clean_text": str(rev["clean_text"]),
                })

        for t in mod["topics"]:
            topics_rows.append({
                "group": "complaint",
                "domain": domain,
                "topic_id": t["topic_id"],
                "topic_label": t["label"],
                "top_terms": "; ".join(t["top_terms"]),
                "top_weights": "; ".join([str(w) for w in t["top_weights"]]),
            })

    assigned_df = pd.concat(assigned_frames, ignore_index=True)
    assigned_df.to_csv(DATA_DIR / "review_topics.csv.gz", index=False, compression=GZIP)
    print(f"  saved review topic assignments -> data/textmining/review_topics.csv.gz")

    pd.DataFrame(exemplars).to_csv(DATA_DIR / "topic_exemplars.csv", index=False)
    print(f"  saved topic exemplars -> data/textmining/topic_exemplars.csv")

    pd.DataFrame(topics_rows).to_csv(DATA_DIR / "topics.csv", index=False)
    print(f"  saved topics table -> data/textmining/topics.csv")

    # Aggregate sentiment by topic and app
    assigned_df["strongly_neg"] = assigned_df["sentiment_compound"] <= -0.5
    assigned_df["neg"] = assigned_df["sentiment_compound"] <= -0.05

    agg_topic = assigned_df.groupby(["domain", "dominant_topic_id", "dominant_topic_label"], observed=True).agg(
        reviews=("review_id", "count"),
        mean_sentiment=("sentiment_compound", "mean"),
        pct_strongly_neg=("strongly_neg", lambda x: round(x.mean() * 100, 2)),
        pct_neg=("neg", lambda x: round(x.mean() * 100, 2)),
    ).reset_index()

    agg_app = assigned_df.groupby(["domain", "dominant_topic_id", "dominant_topic_label", "app_name"], observed=True).agg(
        reviews=("review_id", "count"),
        mean_sentiment=("sentiment_compound", "mean"),
        pct_strongly_neg=("strongly_neg", lambda x: round(x.mean() * 100, 2)),
    ).reset_index()

    agg_topic["mean_sentiment"] = agg_topic["mean_sentiment"].round(4)
    agg_app["mean_sentiment"] = agg_app["mean_sentiment"].round(4)

    agg_topic.to_csv(DATA_DIR / "topic_sentiment.csv", index=False)
    agg_app.to_csv(DATA_DIR / "topic_sentiment_by_app.csv", index=False)
    print(f"  saved sentiment summaries -> data/textmining/topic_sentiment.csv")

    return assigned_df, agg_topic, agg_app


# =============================================================================
# Publication-Quality Charts (tm_02, tm_03, tm_04, tm_05)
# =============================================================================
def generate_charts(complaint_models, praise_models, distinctive_df, agg_topic):
    """Generates the four core publication-quality charts for Issue #134."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    charts_metadata = []

    # Chart 2: Complaint Topics & Top Terms
    fig, axes = plt.subplots(3, 6, figsize=(20, 11))
    for r, domain in enumerate(DOMAINS):
        mod = complaint_models[domain]
        color = DOMAIN_COLORS[domain]
        for c in range(6):
            ax = axes[r, c]
            if c < mod["k"]:
                t = mod["topics"][c]
                terms = t["top_terms"][:8][::-1]
                weights = t["top_weights"][:8][::-1]
                ax.barh(terms, weights, color=color, alpha=0.85, height=0.65)
                ax.set_title(f"T{t['topic_id']}: {t['label']}", fontsize=8.5, fontweight="bold", loc="left", color=INK)
                ax.tick_params(axis="both", labelsize=8)
                ax.grid(axis="y", visible=False)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(domain, fontsize=11, fontweight="bold", color=DOMAIN_COLORS[domain])

    title = "Topic Models (NMF) on 1–2★ Complaints per Domain"
    sub = "Top terms and feature loadings for extracted complaint topics across Food & Grocery (K=6), Shopping (K=6), and Payments (K=5)."
    h = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.text(0.015, 0.98, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.015, 0.95, sub, ha="left", va="top", fontsize=9.5, color=INK2)
    p2 = CHARTS_DIR / "tm_02_complaint_topics.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    print(f"  saved {p2.relative_to(REPO_ROOT)}")
    charts_metadata.append({"file": str(p2.relative_to(REPO_ROOT)), "title": title, "subtitle": sub})

    # Chart 3: Distinctive Terms per App (Monroe et al. Log-Odds)
    fig, axes = plt.subplots(3, 4, figsize=(19, 11))
    col_idx = 0
    row_idx = 0
    current_domain = DOMAINS[0]

    for app in APP_NAMES:
        dom = APP_DOMAIN[app]
        if dom != current_domain:
            row_idx += 1
            col_idx = 0
            current_domain = dom
        ax = axes[row_idx, col_idx]
        app_df = distinctive_df[distinctive_df["app_name"] == app].head(8).iloc[::-1]
        terms = app_df["term"].tolist()
        z_scores = app_df["z_score"].tolist()
        ax.barh(terms, z_scores, color=APP_COLORS[app], height=0.65)
        for y, z in enumerate(z_scores):
            ax.text(z + 0.5, y, f"{z:.1f}", va="center", fontsize=7.5, color=INK2)
        ax.set_title(f"{app} ({dom})", fontsize=9.5, fontweight="bold", loc="left", color=INK)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(z_scores) * 1.25 if z_scores else 10)
        col_idx += 1

    # Turn off the empty slot in row 2 (Payments has 3 apps)
    axes[2, 3].axis("off")

    title = "Most Distinctive Words & Phrases per App (Standardized Log-Odds Ratio)"
    sub = "Top terms differentiating each app from others within its domain, calculated via Monroe et al. (2008) log-odds with informative Dirichlet prior."
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.text(0.015, 0.98, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.015, 0.95, sub, ha="left", va="top", fontsize=9.5, color=INK2)
    p3 = CHARTS_DIR / "tm_03_distinctive_terms.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    print(f"  saved {p3.relative_to(REPO_ROOT)}")
    charts_metadata.append({"file": str(p3.relative_to(REPO_ROOT)), "title": title, "subtitle": sub})

    # Chart 4: Praise Topics as a Contrast (4–5★ Reviews)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for r, domain in enumerate(DOMAINS):
        mod = praise_models[domain]
        for c in range(4):
            ax = axes[r, c]
            t = mod["topics"][c]
            terms = t["top_terms"][:7][::-1]
            weights = t["top_weights"][:7][::-1]
            ax.barh(terms, weights, color="#34a853", alpha=0.75, height=0.65)
            ax.set_title(f"{t['label']}", fontsize=8.5, fontweight="bold", loc="left", color=INK)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(axis="y", visible=False)
        axes[r, 0].set_ylabel(domain, fontsize=10.5, fontweight="bold", color=DOMAIN_COLORS[domain])

    title = "Topics of Praise (4–5★ Reviews) as a Contrast: What Satisfied Users Mention"
    sub = "NMF topic models fitted on 4–5★ praise reviews across Food & Grocery, Shopping, and Payments (K=4)."
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.text(0.015, 0.98, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.015, 0.95, sub, ha="left", va="top", fontsize=9.5, color=INK2)
    p4 = CHARTS_DIR / "tm_04_praise_topics.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    print(f"  saved {p4.relative_to(REPO_ROOT)}")
    charts_metadata.append({"file": str(p4.relative_to(REPO_ROOT)), "title": title, "subtitle": sub})

    # Chart 5: Topic Sentiment & Severity Hierarchy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    df_sorted = agg_topic.sort_values("mean_sentiment").reset_index(drop=True)
    topic_names = [f"{r['domain']}: {r['dominant_topic_label']}" for _, r in df_sorted.iterrows()]
    colors = [DOMAIN_COLORS[r["domain"]] for _, r in df_sorted.iterrows()]

    ax1.barh(topic_names, df_sorted["mean_sentiment"], color=colors, height=0.65)
    for y, v in enumerate(df_sorted["mean_sentiment"]):
        ax1.text(v - 0.012, y, f"{v:.3f}", va="center", ha="right", fontsize=8, color=INK)
    ax1.set_xlim(-0.75, 0.02)
    ax1.set_xlabel("Mean VADER Compound Sentiment Score")
    ax1.set_title("Average Sentiment Score by Complaint Topic (Most Negative First)", loc="left", fontweight="bold", fontsize=10.5)
    ax1.grid(axis="y", visible=False)

    df_neg_sorted = agg_topic.sort_values("pct_strongly_neg", ascending=False).reset_index(drop=True)
    topic_names_neg = [f"{r['domain']}: {r['dominant_topic_label']}" for _, r in df_neg_sorted.iterrows()]
    colors_neg = [DOMAIN_COLORS[r["domain"]] for _, r in df_neg_sorted.iterrows()]

    ax2.barh(topic_names_neg[::-1], df_neg_sorted["pct_strongly_neg"][::-1], color=colors_neg[::-1], height=0.65)
    for y, v in enumerate(df_neg_sorted["pct_strongly_neg"][::-1]):
        ax2.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of Reviews with Severe Negative Sentiment (Compound ≤ -0.5)")
    ax2.set_title("Proportion of Strongly Negative Reviews by Topic", loc="left", fontweight="bold", fontsize=10.5)
    ax2.grid(axis="y", visible=False)

    title = "Complaint Topic Sentiment Severity Hierarchy Across Domains"
    sub = "Comparison of average VADER polarity and share of strongly negative reviews (compound ≤ -0.5) across all 17 complaint topics."
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.text(0.015, 0.98, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.015, 0.95, sub, ha="left", va="top", fontsize=9.5, color=INK2)
    p5 = CHARTS_DIR / "tm_05_topic_sentiment.png"
    fig.savefig(p5, dpi=160)
    plt.close(fig)
    print(f"  saved {p5.relative_to(REPO_ROOT)}")
    charts_metadata.append({"file": str(p5.relative_to(REPO_ROOT)), "title": title, "subtitle": sub})

    return charts_metadata


# =============================================================================
# Main Routine & CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 2 Text Mining: Topic Modelling & Distinctive Terms")
    parser.add_argument("--predict", type=str, help="Predict topic for a review text string")
    parser.add_argument("--domain", type=str, default="Food & Grocery", choices=DOMAINS, help="Domain for prediction")
    args = parser.parse_args()

    # Interactive prediction mode
    if args.predict:
        slug = DOMAIN_SLUG[args.domain]
        model_path = MODELS_DIR / f"topic_pipeline_complaint_{slug}.pkl"
        if not model_path.exists():
            sys.exit(f"Model not found at {model_path}. Run without arguments first to fit models.")
        pipeline = joblib.load(model_path)
        pred = pipeline.predict(args.predict)
        print(f"\nReview: '{args.predict}'")
        print(f"Domain: {args.domain}")
        print(f"Dominant Topic: {pred['dominant_topic']} (Topic {pred['dominant_topic_id']}, confidence: {pred['confidence']:.2%})")
        print("Topic distribution:")
        for t, p in pred["weights"].items():
            print(f"  - {t}: {p:.2%}")
        return

    print("=" * 70)
    print("Stage 2 Text Mining: Topic Modelling & Distinctive Terms per App (#134)")
    print("=" * 70)

    # 1. Fit Complaint Topic Models
    complaint_models = fit_topic_models("complaint")

    # 2. Fit Praise Contrast Models
    praise_models = fit_topic_models("praise")

    # 3. Compute Distinctive Terms per App
    distinctive_df = compute_distinctive_terms()

    # 4. Evaluate Topics, Exemplars and Sentiment
    assigned_df, agg_topic, agg_app = evaluate_complaint_topics(complaint_models)

    # 5. Generate Charts
    charts_metadata = generate_charts(complaint_models, praise_models, distinctive_df, agg_topic)

    # 6. Save JSON Summary
    summary = {
        "status": "completed",
        "stage": "Review 2, Stage 2 (Issue #134)",
        "models_saved": [str(p.relative_to(REPO_ROOT)) for p in MODELS_DIR.glob("*.pkl")],
        "complaint_topics": {
            d: [
                {"id": t["topic_id"], "label": t["label"], "top_terms": t["top_terms"][:10]}
                for t in complaint_models[d]["topics"]
            ]
            for d in DOMAINS
        },
        "praise_topics": {
            d: [
                {"id": t["topic_id"], "label": t["label"], "top_terms": t["top_terms"][:8]}
                for t in praise_models[d]["topics"]
            ]
            for d in DOMAINS
        },
        "sentiment_by_topic": agg_topic.to_dict(orient="records"),
        "top_distinctive_terms_by_app": {
            a: distinctive_df[distinctive_df["app_name"] == a].head(8)[["term", "z_score", "app_count"]].to_dict(orient="records")
            for a in APP_NAMES
        },
        "charts": charts_metadata,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved comprehensive summary to {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print("Done! Issue #134 implementation and evidence generation complete.")


if __name__ == "__main__":
    main()
