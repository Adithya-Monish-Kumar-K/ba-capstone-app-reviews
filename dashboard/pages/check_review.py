
"""Interactive review checker: problem-review probability, complaint topic and VADER sentiment."""

import importlib
import runpy
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
TOPIC_DIR = MODEL_DIR / "textmining"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

tagger = importlib.import_module("03_tag_issues")

# Support legacy topic models saved when TopicPipeline belonged to __main__.
_topic_script = runpy.run_path(
    str(SCRIPTS_DIR / "17_topic_modelling.py"),
    run_name="topic_modelling_compat",
)
sys.modules["__main__"].TopicPipeline = _topic_script["TopicPipeline"]

ISSUE_COLS = [f"issue_{c}" for c in tagger.TAXONOMY]
SENTIMENT_COLS = [
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_pos",
    "sentiment_compound",
]
STRUCTURAL_COLS = ISSUE_COLS + SENTIMENT_COLS + [
    "review_length",
    "thumbs_up",
]

DOMAIN_MODELS = {
    "Food & Grocery": "food_grocery",
    "Payments": "payments",
    "Shopping": "shopping",
}

EXAMPLES = {
    "Food & Grocery": [
        "My grocery order arrived late and several items were missing.",
        "The delivery partner never arrived and support did not help.",
        "Very easy to order groceries and the delivery was quick.",
    ],
    "Payments": [
        "Money was deducted but my payment is still pending.",
        "My refund has not arrived even after several days.",
        "Payments are fast and the transaction history is useful.",
    ],
    "Shopping": [
        "The app crashes whenever I try to place an order.",
        "I returned the product but have not received my refund.",
        "The shopping experience was smooth and delivery was on time.",
    ],
}


@st.cache_resource
def load_models():
    names = [
        "tfidf_vectorizer",
        "svd_model",
        "structural_scaler",
        "logistic_regression",
        "random_forest",
    ]
    return {
        name: joblib.load(MODEL_DIR / f"{name}.pkl")
        for name in names
    }


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_topic_pipeline(slug):
    return joblib.load(
        TOPIC_DIR / f"topic_pipeline_complaint_{slug}.pkl"
    )


def select_example(example):
    """Queue an example to populate the review box on the next rerun."""
    st.session_state["pending_review_example"] = example


def predict_problem(text, models, sia):
    """Build training-compatible features and predict low-star probabilities."""
    tags = tagger.tag_review_text(text)
    sentiment = sia.polarity_scores(text)

    row = {
        **{col: tags[col] for col in ISSUE_COLS},
        "sentiment_neg": sentiment["neg"],
        "sentiment_neu": sentiment["neu"],
        "sentiment_pos": sentiment["pos"],
        "sentiment_compound": sentiment["compound"],
        "review_length": len(text.split()),
        "thumbs_up": 0,
    }

    structural = pd.DataFrame([row])[STRUCTURAL_COLS].astype(float)
    tfidf = models["tfidf_vectorizer"].transform([text])
    text_features = models["svd_model"].transform(tfidf)

    X = np.hstack(
        [
            text_features,
            models["structural_scaler"].transform(structural),
        ]
    ).astype(np.float32)

    probabilities = {
        "Logistic Regression": float(
            models["logistic_regression"].predict_proba(X)[0, 1]
        ),
        "Random Forest": float(
            models["random_forest"].predict_proba(X)[0, 1]
        ),
    }

    detected_tags = [
        col.removeprefix("issue_")
        for col in ISSUE_COLS
        if tags[col]
    ]

    return probabilities, sentiment, detected_tags


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.title("Check a Review")
st.caption(
    "Explore whether a review resembles reviews associated with low star "
    "ratings, which complaint topic it matches, and its VADER sentiment. "
    "A prediction is not confirmation of a verified app failure."
)

domain = st.selectbox(
    "Choose the review's app domain",
    list(DOMAIN_MODELS),
    key="check_review_domain",
)

# Initialise the text-area state before creating the widget.
if "check_review_text" not in st.session_state:
    st.session_state["check_review_text"] = ""

# Apply a queued example before the text-area widget is instantiated.
if "pending_review_example" in st.session_state:
    st.session_state["check_review_text"] = st.session_state.pop(
        "pending_review_example"
    )

review = st.text_area(
    "Paste a review",
    key="check_review_text",
    placeholder=(
        "Example: Money was deducted but my order failed "
        "and I never received a refund."
    ),
    height=150,
)

with st.expander("Try an example review"):
    for i, example in enumerate(EXAMPLES[domain]):
        st.button(
            example,
            key=f"example_{domain}_{i}",
            use_container_width=True,
            on_click=select_example,
            args=(example,),
        )

threshold = st.slider(
    "Problem-review probability threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help=(
        "A review is flagged when its predicted probability "
        "reaches this threshold."
    ),
)

if st.button(
    "Analyze review",
    type="primary",
    use_container_width=True,
):
    text = review.strip()

    if not text:
        st.warning("Please enter a review first.")
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                models = load_models()
                sia = load_vader()

                probabilities, sentiment, tags = predict_problem(
                    text, models, sia
                )

                topic_model = load_topic_pipeline(
                    DOMAIN_MODELS[domain]
                )
                topic = topic_model.predict(text)

            st.subheader("Problem-review prediction")
            st.caption(
                "The target is a likely 1–2 star rating, "
                "not a verified software failure."
            )

            cols = st.columns(2)

            for col, (name, probability) in zip(
                cols, probabilities.items()
            ):
                with col:
                    st.metric(name, f"{probability:.1%}")

                    if probability >= threshold:
                        st.error(
                            "Flagged as a likely problem review"
                        )
                    else:
                        st.success(
                            "Below the selected threshold"
                        )

                    st.progress(probability)

            st.subheader("Complaint topic")

            if topic["dominant_topic_id"] == 0:
                st.info(
                    "This review could not be assigned "
                    "to a complaint topic."
                )
            else:
                st.write(f"**{topic['dominant_topic']}**")
                st.caption(
                    f"Relative topic weight: "
                    f"{topic['confidence']:.1%}. "
                    "This is the dominant topic weight, "
                    "not a calibrated probability."
                )

            st.subheader("VADER sentiment")

            c = st.columns(4)
            c[0].metric("Negative", f"{sentiment['neg']:.2f}")
            c[1].metric("Neutral", f"{sentiment['neu']:.2f}")
            c[2].metric("Positive", f"{sentiment['pos']:.2f}")
            c[3].metric("Compound", f"{sentiment['compound']:+.2f}")

            st.caption(
                "Negative, neutral and positive scores range from "
                "0 to 1. The compound score ranges from −1 to +1. "
                "VADER is lexicon-based and may miss contextual meaning."
            )

            st.subheader("Detected issue tags")
            st.write(
                ", ".join(tags)
                if tags
                else "No keyword issue tags detected."
            )

        except LookupError:
            st.error(
                "The VADER lexicon is missing. Run this in Terminal: "
                "python -c \"import nltk; "
                "nltk.download('vader_lexicon')\""
            )
        except Exception as exc:
            st.error(f"Could not analyze this review: {exc}")
