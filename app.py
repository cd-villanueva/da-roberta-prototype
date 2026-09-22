from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime

import streamlit as st


APP_TITLE = "DA-RoBERTa Bias Detection Prototype"
FIXED_PROTOTYPE_SEED = 2026

SAMPLE_SENTENCES = {
    "Neutral example": (
        "The committee released its report on Tuesday after a three-month review."
    ),
    "Potentially biased example": (
        '"We have one beautiful law," Trump recently said in his '
        "characteristically bizarre syntax and diction."
    ),
    "WNC original": "most of the gameplay is pilfered from ddr.",
    "WNC neutral revision": "most of the gameplay is based on ddr.",
}


@dataclass(frozen=True)
class PrototypePrediction:
    model: str
    label: str
    confidence: float
    non_biased_probability: float
    biased_probability: float
    non_biased_logit: float
    biased_logit: float
    tokens: list[str]
    representation_preview: list[float]


def stable_seed(text: str, model_key: str) -> int:
    """Create a repeatable integer seed without using Python's salted hash."""
    payload = f"{FIXED_PROTOTYPE_SEED}|{model_key}|{text.strip()}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def illustrative_tokens(text: str) -> list[str]:
    """Return a readable token preview; this is not the RoBERTa tokenizer."""
    pieces = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\w\s]", text)
    return ["<s>", *pieces[:30], "</s>"]


def softmax_pair(first: float, second: float) -> tuple[float, float]:
    maximum = max(first, second)
    first_exp = math.exp(first - maximum)
    second_exp = math.exp(second - maximum)
    denominator = first_exp + second_exp
    return first_exp / denominator, second_exp / denominator


def create_seeded_prediction(text: str, model_key: str) -> PrototypePrediction:
    """Generate deterministic, simulated model values for UI demonstration."""
    generator = random.Random(stable_seed(text, model_key))

    # These are seeded demonstration values, not outputs from trained weights.
    center = generator.uniform(-0.55, 0.55)
    separation = generator.uniform(-1.35, 1.35)
    if model_key == "enhanced":
        separation *= 1.12

    non_biased_logit = center - separation / 2
    biased_logit = center + separation / 2
    non_biased_probability, biased_probability = softmax_pair(
        non_biased_logit,
        biased_logit,
    )

    if biased_probability > non_biased_probability:
        label = "Biased"
        confidence = biased_probability
    else:
        label = "Non-biased"
        confidence = non_biased_probability

    representation_preview = [
        round(generator.uniform(-1.0, 1.0), 3) for _ in range(6)
    ]

    display_name = (
        "Baseline DA-RoBERTa"
        if model_key == "baseline"
        else "Enhanced DA-RoBERTa"
    )

    return PrototypePrediction(
        model=display_name,
        label=label,
        confidence=confidence,
        non_biased_probability=non_biased_probability,
        biased_probability=biased_probability,
        non_biased_logit=non_biased_logit,
        biased_logit=biased_logit,
        tokens=illustrative_tokens(text),
        representation_preview=representation_preview,
    )


def probability_bar(label: str, probability: float, color_class: str) -> None:
    percent = probability * 100
    st.markdown(
        f"""
        <div class="probability-row">
            <div class="probability-label">
                <span>{label}</span><strong>{percent:.1f}%</strong>
            </div>
            <div class="probability-track">
                <div class="probability-fill {color_class}" style="width:{percent:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(prediction: PrototypePrediction) -> None:
    label_class = "biased-label" if prediction.label == "Biased" else "neutral-label"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-eyebrow">SIMULATED MODEL VIEW</div>
            <h3>{prediction.model}</h3>
            <div class="prediction-line">
                <span class="prediction-label {label_class}">{prediction.label}</span>
                <span class="confidence-text">{prediction.confidence * 100:.1f}% confidence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    probability_bar(
        "Non-biased",
        prediction.non_biased_probability,
        "neutral-fill",
    )
    probability_bar(
        "Biased",
        prediction.biased_probability,
        "biased-fill",
    )

    with st.expander("View illustrative processing details"):
        st.markdown("**1. Input sentence**")
        st.write(st.session_state.current_sentence)

        st.markdown("**2. Token preview**")
        st.code(" ".join(prediction.tokens), language=None)
        st.caption(
            "This readable preview illustrates tokenization; it is not the actual "
            "RoBERTa tokenizer output."
        )

        st.markdown("**3. Encoder representation preview**")
        st.code(str(prediction.representation_preview), language=None)
        st.caption("Six illustrative values are shown from a conceptual 768-value representation.")

        st.markdown("**4. Two class logits**")
        logit_col_1, logit_col_2 = st.columns(2)
        logit_col_1.metric("Non-biased logit", f"{prediction.non_biased_logit:.3f}")
        logit_col_2.metric("Biased logit", f"{prediction.biased_logit:.3f}")

        st.markdown("**5. Softmax and decision**")
        st.write(
            "Softmax converts the two logits into probabilities. The class with "
            "the larger probability becomes the displayed prediction."
        )


def prediction_payload(
    sentence: str,
    predictions: list[PrototypePrediction],
) -> str:
    data = {
        "prototype_mode": True,
        "disclaimer": "Values are seeded simulations and are not experimental results.",
        "sentence": sentence,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "predictions": [asdict(prediction) for prediction in predictions],
    }
    return json.dumps(data, indent=2)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #13233a;
            --muted: #607086;
            --paper: #ffffff;
            --blue: #2e6f95;
            --blue-soft: #e8f3f8;
            --rose: #a8495b;
            --rose-soft: #faeaee;
            --gold: #d9aa45;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 5%, rgba(217,170,69,.10), transparent 28rem),
                linear-gradient(180deg, #f8fbfd 0%, #f3f6f9 100%);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2.3rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
        .hero {
            padding: 2.1rem 2.2rem;
            border: 1px solid rgba(19,35,58,.10);
            border-radius: 24px;
            background: rgba(255,255,255,.88);
            box-shadow: 0 18px 50px rgba(31,55,80,.08);
            margin-bottom: 1.4rem;
        }
        .hero-kicker, .result-eyebrow {
            color: var(--blue);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .14em;
        }
        .hero h1 { margin: .4rem 0 .6rem; font-size: clamp(2rem, 5vw, 3.35rem); }
        .hero p { color: var(--muted); font-size: 1.03rem; max-width: 780px; margin: 0; }
        .prototype-notice {
            border-left: 4px solid var(--gold);
            background: #fff8e8;
            color: #5f4a1e;
            padding: .9rem 1rem;
            border-radius: 0 12px 12px 0;
            margin: 1rem 0 1.5rem;
        }
        .result-card {
            margin-top: .5rem;
            padding: 1.25rem 1.35rem;
            border-radius: 18px;
            background: var(--paper);
            border: 1px solid rgba(19,35,58,.11);
            box-shadow: 0 10px 30px rgba(31,55,80,.06);
        }
        .result-card h3 { margin: .25rem 0 .8rem; }
        .prediction-line { display: flex; align-items: center; gap: .7rem; flex-wrap: wrap; }
        .prediction-label {
            display: inline-flex;
            padding: .36rem .72rem;
            border-radius: 999px;
            font-weight: 800;
        }
        .biased-label { color: #862f42; background: var(--rose-soft); }
        .neutral-label { color: #1f6284; background: var(--blue-soft); }
        .confidence-text { color: var(--muted); font-size: .92rem; }
        .probability-row { margin: .9rem .15rem; }
        .probability-label { display:flex; justify-content:space-between; color:var(--ink); margin-bottom:.35rem; }
        .probability-track { height: 10px; overflow:hidden; background:#e6ebef; border-radius:999px; }
        .probability-fill { height:100%; border-radius:999px; }
        .neutral-fill { background: linear-gradient(90deg, #63a9ca, #2e6f95); }
        .biased-fill { background: linear-gradient(90deg, #d87a8c, #a8495b); }
        .flow {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: .6rem;
            align-items: center;
            margin: 1.2rem 0 1.8rem;
        }
        .flow-step {
            min-height: 88px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: .75rem;
            background: white;
            border: 1px solid rgba(19,35,58,.11);
            border-radius: 14px;
            color: var(--ink);
            font-weight: 700;
        }
        .flow-step small { display:block; color:var(--muted); font-weight:500; margin-top:.25rem; }
        div[data-testid="stSidebar"] { background: #eef4f7; }
        @media (max-width: 760px) {
            .flow { grid-template-columns: 1fr; }
            .hero { padding: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📰",
    layout="wide",
)
inject_styles()

if "current_sentence" not in st.session_state:
    st.session_state.current_sentence = SAMPLE_SENTENCES["Potentially biased example"]
if "predictions" not in st.session_state:
    st.session_state.predictions = []

with st.sidebar:
    st.markdown("## Prototype controls")
    model_view = st.radio(
        "Model view",
        ["Compare both", "Baseline only", "Enhanced only"],
        help="The enhanced view is simulated and does not load trained enhanced weights.",
    )
    st.markdown("---")
    st.markdown("**Prototype status**")
    st.success("Seeded demonstration mode")
    st.caption(f"Fixed demonstration seed: {FIXED_PROTOTYPE_SEED}")
    st.caption("No accuracy, F1, or experimental performance is claimed by this app.")

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">SENTENCE-LEVEL MEDIA BIAS</div>
        <h1>How a news sentence becomes a prediction</h1>
        <p>
            A deployable interface prototype for demonstrating the intended
            DA-RoBERTa input, processing stages, class scores, and output.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="prototype-notice">
        <strong>Prototype demonstration:</strong> All displayed logits and probabilities
        are deterministic seeded values. They are not predictions or test results from the
        trained baseline or proposed enhanced model.
    </div>
    """,
    unsafe_allow_html=True,
)

analyzer_tab, process_tab, study_tab = st.tabs(
    ["Analyze sentence", "How the system works", "About the study"]
)

with analyzer_tab:
    st.subheader("Enter one English news sentence")
    sample_choice = st.selectbox(
        "Load an example",
        ["Keep current text", *SAMPLE_SENTENCES.keys()],
    )
    if sample_choice != "Keep current text":
        st.session_state.current_sentence = SAMPLE_SENTENCES[sample_choice]

    sentence = st.text_area(
        "News sentence",
        key="current_sentence",
        height=145,
        max_chars=1500,
        placeholder="Paste or type one English news sentence here...",
    )

    analyze_clicked = st.button(
        "Generate prototype prediction",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            st.warning("Enter a sentence before generating a prediction.")
            st.session_state.predictions = []
        else:
            model_keys = {
                "Compare both": ["baseline", "enhanced"],
                "Baseline only": ["baseline"],
                "Enhanced only": ["enhanced"],
            }[model_view]
            st.session_state.predictions = [
                create_seeded_prediction(cleaned_sentence, key) for key in model_keys
            ]

    if st.session_state.predictions:
        st.markdown("### Prototype output")
        result_columns = st.columns(len(st.session_state.predictions))
        for column, prediction in zip(result_columns, st.session_state.predictions):
            with column:
                result_card(prediction)

        st.download_button(
            "Download this simulated output as JSON",
            data=prediction_payload(sentence, st.session_state.predictions),
            file_name="prototype_bias_output.json",
            mime="application/json",
            use_container_width=True,
        )

with process_tab:
    st.subheader("Illustrative processing pipeline")
    st.markdown(
        """
        <div class="flow">
            <div class="flow-step">1. Sentence input<small>One English news sentence</small></div>
            <div class="flow-step">2. Tokenization<small>Input IDs and attention mask</small></div>
            <div class="flow-step">3. RoBERTa encoder<small>Contextual representation</small></div>
            <div class="flow-step">4. Classification head<small>Two class logits</small></div>
            <div class="flow-step">5. Decision<small>Softmax probabilities and label</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        The intended trained system accepts one sentence at prediction time. The
        RoBERTa tokenizer converts it into input IDs and an attention mask. The
        encoder produces a contextual representation, and the classification head
        produces two logits: one for **non-biased** and one for **biased**. Softmax
        converts the logits into probabilities, and the class with the larger value
        becomes the prediction.

        The proposed enhanced model uses pairwise-ranking loss only during WNC
        training. A neutral counterpart is not required when a user submits a
        sentence to the final interface.
        """
    )

with study_tab:
    st.subheader("Prototype boundaries")
    st.markdown(
        """
        **Included in this prototype**

        - Single-sentence text input
        - Baseline and enhanced-model interface views
        - Illustrative tokens, representation values, logits, probabilities, and label
        - Repeatable seeded outputs for demonstrations
        - Downloadable simulated result

        **Not included in this prototype**

        - Loading the trained baseline checkpoint
        - Training or loading an enhanced-model checkpoint
        - Real model inference or experimental evaluation
        - Accuracy, macro-F1, or other performance claims

        The production version can later replace the seeded prediction function with
        real checkpoint inference while preserving the same user-interface workflow.
        """
    )

