"""Compare NMF with LDA on the same complaint matrices (Review 2, Method 1 evaluation).

Both models get the same documents, the same vocabulary and the same number of topics (the K
chosen in scripts/18_topic_evaluation.py). Each is fitted on 80% of a domain's complaint reviews,
and its topics are scored on the other 20%, which neither model saw:

- held-out NPMI: how often each topic's top 10 terms appear together in unseen reviews (-1 to 1);
- topic diversity: share of distinct terms among all topics' top 10 terms (1 = no overlap);
- largest topic share: share of held-out reviews whose dominant topic is the biggest topic
  (a model that pours most reviews into one topic separates the complaints less);
- fit time in seconds.

NMF works on TF-IDF weights, LDA on raw counts, as each method expects.

Outputs: data/textmining/topic_model_comparison.csv, data/charts/textmining/tm_10_nmf_vs_lda.png
Run:     python scripts/23_topic_model_comparison.py   (about 3 minutes)
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from apps import DOMAINS  # noqa: E402

DTM_DIR = ROOT / "data" / "textmining" / "dtm"
OUT_CSV = ROOT / "data" / "textmining" / "topic_model_comparison.csv"
OUT_PNG = ROOT / "data" / "charts" / "textmining" / "tm_10_nmf_vs_lda.png"

DOMAIN_SLUG = {"Food & Grocery": "food_grocery", "Shopping": "shopping", "Payments": "payments"}
CHOSEN_K = {"Food & Grocery": 6, "Shopping": 6, "Payments": 5}
TOP_N = 10
SEED = 42


def npmi(components, X_bin):
    """Mean NPMI of each topic's top-N terms, counted on the documents in X_bin."""
    n = X_bin.shape[0]
    scores = []
    for comp in components:
        top = comp.argsort()[::-1][:TOP_N]
        sub = X_bin[:, top].toarray()
        co = sub.T @ sub
        d = np.diag(co)
        pairs = []
        for i in range(1, TOP_N):
            for j in range(i):
                p_ij = (co[i, j] + 1e-12) / n
                p_i, p_j = (d[i] + 1e-12) / n, (d[j] + 1e-12) / n
                pairs.append(np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij)))
        scores.append(np.mean(pairs))
    return float(np.mean(scores))


def diversity(components):
    tops = [comp.argsort()[::-1][:TOP_N] for comp in components]
    return len(set(np.concatenate(tops))) / (TOP_N * len(tops))


def largest_share(doc_topic):
    dominant = doc_topic.argmax(axis=1)
    return float(np.bincount(dominant).max() / len(dominant))


def main():
    rows = []
    for domain in DOMAINS:
        X = sparse.load_npz(DTM_DIR / f"complaint_{DOMAIN_SLUG[domain]}.npz").tocsr()
        idx_train, idx_test = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=SEED)
        X_train, X_test = X[idx_train], X[idx_test]
        X_test_bin = (X_test > 0).astype(np.float32)
        k = CHOSEN_K[domain]

        tfidf = TfidfTransformer().fit(X_train)
        t0 = time.time()
        nmf = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=200).fit(tfidf.transform(X_train))
        nmf_time = time.time() - t0

        t0 = time.time()
        lda = LatentDirichletAllocation(n_components=k, learning_method="online", batch_size=4096,
                                        max_iter=10, random_state=SEED, n_jobs=1).fit(X_train)
        lda_time = time.time() - t0

        for name, model, doc_topic in [
            ("NMF (TF-IDF)", nmf, nmf.transform(tfidf.transform(X_test))),
            ("LDA (counts)", lda, lda.transform(X_test)),
        ]:
            rows.append({
                "domain": domain, "model": name, "k": k,
                "train_reviews": X_train.shape[0], "test_reviews": X_test.shape[0],
                "heldout_npmi": round(npmi(model.components_, X_test_bin), 4),
                "topic_diversity": round(diversity(model.components_), 3),
                "largest_topic_share": round(largest_share(doc_topic), 3),
                "fit_seconds": round(nmf_time if name.startswith("NMF") else lda_time, 1),
            })
            print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_CSV.relative_to(ROOT)}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    colors = {"NMF (TF-IDF)": "#2a78d6", "LDA (counts)": "#9aa5b1"}
    for ax, (col, title) in zip(axes, [("heldout_npmi", "Held-out NPMI (higher is better)"),
                                       ("topic_diversity", "Topic diversity (higher is better)"),
                                       ("largest_topic_share", "Largest topic's share of reviews")]):
        x = np.arange(len(DOMAINS))
        for i, m in enumerate(colors):
            vals = [df[(df.domain == d) & (df.model == m)][col].iloc[0] for d in DOMAINS]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=colors[m], label=m)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, DOMAINS, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("NMF vs LDA on the same complaint reviews, same K, scored on a 20% held-out split",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f"saved {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
