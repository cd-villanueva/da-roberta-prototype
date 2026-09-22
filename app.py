from __future__ import annotations

import gc
import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, RobertaConfig, RobertaModel


APP_TITLE = "DA-RoBERTa Bias Detection Prototype"
HF_REPO_ID = "clarkkyvii/da-roberta-baseline-prototype"
HF_BUNDLE_FILENAME = "streamlit_baseline_bundle.zip"
MAX_LENGTH = 512

SAMPLE_SENTENCES = {
    "Neutral example": "The committee released its report on Tuesday after a three-month review.",
    "Potentially biased example": (
        '"We have one beautiful law," Trump recently said in his '
        "characteristically bizarre syntax and diction."
    ),
    "WNC original": "most of the gameplay is pilfered from ddr.",
    "WNC neutral revision": "most of the gameplay is based on ddr.",
}


@dataclass(frozen=True)
class BaselinePrediction:
    model: str
    label: str
    confidence: float
    non_biased_probability: float
    biased_probability: float
    non_biased_logit: float
    biased_logit: float
    tokens: list[str]
    representation_preview: list[float]


class DARobertaClassifier(nn.Module):
    """Architecture used by the WNC and BABE baseline notebooks."""

    accepts_loss_kwargs = False

    def __init__(self) -> None:
        super().__init__()
        config = RobertaConfig(
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=514,
            type_vocab_size=1,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            layer_norm_eps=1e-5,
        )
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.classifier1 = nn.Linear(config.hidden_size, 2)

    def logits_and_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sentence_representation = encoder_output.last_hidden_state[:, 0, :]
        transformed = self.vocab_transform(sentence_representation)
        logits = self.classifier1(self.dropout(transformed))
        return logits, transformed


@st.cache_resource(show_spinner="Downloading and loading the trained baseline model...")
def load_baseline() -> tuple[DARobertaClassifier, object, dict]:
    zip_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_BUNDLE_FILENAME,
            repo_type="model",
        )
    )
    extract_root = Path(tempfile.gettempdir()) / "da_roberta_fold1_bundle_v1"
    manifest_path = extract_root / "model_manifest.json"

    if not manifest_path.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as bundle:
            bundle.extractall(extract_root)

    model_path = extract_root / "baseline_babe_fold_1_model_state.pt"
    tokenizer_path = extract_root / "tokenizer"
    required_paths = [model_path, tokenizer_path / "tokenizer.json", manifest_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"The deployment bundle is incomplete: {missing_paths}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = DARobertaClassifier()

    try:
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    try:
        load_result = model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        load_result = model.load_state_dict(state_dict, strict=True)

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    model.eval()
    del state_dict
    gc.collect()

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    return model, tokenizer, manifest


def predict_with_baseline(text: str) -> BaselinePrediction:
    model, tokenizer, _ = load_baseline()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    with torch.inference_mode():
        logits, representation = model.logits_and_representation(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        probabilities = torch.softmax(logits, dim=-1)

    non_biased_probability = float(probabilities[0, 0].item())
    biased_probability = float(probabilities[0, 1].item())
    non_biased_logit = float(logits[0, 0].item())
    biased_logit = float(logits[0, 1].item())
    predicted_label = int(probabilities.argmax(dim=-1).item())
    label = "Biased" if predicted_label == 1 else "Non-biased"
    confidence = max(non_biased_probability, biased_probability)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
    representation_preview = [
        round(float(value), 4) for value in representation[0, :6].tolist()
    ]

    return BaselinePrediction(
        model="Baseline DA-RoBERTa — BABE Fold 1",
        label=label,
        confidence=confidence,
        non_biased_probability=non_biased_probability,
        biased_probability=biased_probability,
        non_biased_logit=non_biased_logit,
        biased_logit=biased_logit,
        tokens=tokens,
        representation_preview=representation_preview,
    )


def probability_bar(label: str, probability: float, color_class: str) -> None:
    percent = probability * 100
    st.markdown(
        f"""
        <div class="probability-row">
            <div class="probability-label"><span>{label}</span><strong>{percent:.1f}%</strong></div>
            <div class="probability-track">
                <div class="probability-fill {color_class}" style="width:{percent:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(prediction: BaselinePrediction) -> None:
    label_class = "biased-label" if prediction.label == "Biased" else "neutral-label"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-eyebrow">REAL TRAINED BASELINE INFERENCE</div>
            <h3>{prediction.model}</h3>
            <div class="prediction-line">
                <span class="prediction-label {label_class}">{prediction.label}</span>
                <span class="confidence-text">{prediction.confidence * 100:.1f}% model confidence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    probability_bar("Non-biased", prediction.non_biased_probability, "neutral-fill")
    probability_bar("Biased", prediction.biased_probability, "biased-fill")

    with st.expander("View actual processing details"):
        st.markdown("**1. Input sentence**")
        st.write(st.session_state.current_sentence)
        st.markdown("**2. RoBERTa tokens**")
        st.code(" ".join(prediction.tokens), language=None)
        st.caption(f"Actual tokenizer output: {len(prediction.tokens)} tokens.")
        st.markdown("**3. Transformed sentence representation preview**")
        st.code(str(prediction.representation_preview), language=None)
        st.caption("The first six values of the actual 768-value transformed representation.")
        st.markdown("**4. Two class logits**")
        logit_col_1, logit_col_2 = st.columns(2)
        logit_col_1.metric("Non-biased logit", f"{prediction.non_biased_logit:.4f}")
        logit_col_2.metric("Biased logit", f"{prediction.biased_logit:.4f}")
        st.markdown("**5. Softmax and decision**")
        st.write(
            "Softmax converts the two real logits into probabilities. The class "
            "with the larger probability becomes the displayed prediction."
        )


def prediction_payload(sentence: str, prediction: BaselinePrediction) -> str:
    return json.dumps(
        {
            "inference_mode": "real_trained_baseline",
            "checkpoint_scope": "BABE Fold 1 prototype checkpoint",
            "model_repository": HF_REPO_ID,
            "sentence": sentence,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "prediction": asdict(prediction),
        },
        indent=2,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#13233a; --muted:#607086; --paper:#fff; --blue:#2e6f95;
                --blue-soft:#e8f3f8; --rose:#a8495b; --rose-soft:#faeaee; --gold:#d9aa45; }
        .stApp { background:radial-gradient(circle at 8% 5%,rgba(217,170,69,.10),transparent 28rem),
                 linear-gradient(180deg,#f8fbfd 0%,#f3f6f9 100%); }
        .block-container { max-width:1120px; padding-top:2.3rem; padding-bottom:4rem; }
        h1,h2,h3 { color:var(--ink); letter-spacing:-.02em; }
        .hero { padding:2.1rem 2.2rem; border:1px solid rgba(19,35,58,.10); border-radius:24px;
                background:rgba(255,255,255,.88); box-shadow:0 18px 50px rgba(31,55,80,.08);
                margin-bottom:1.4rem; }
        .hero-kicker,.result-eyebrow { color:var(--blue); font-size:.72rem; font-weight:800; letter-spacing:.14em; }
        .hero h1 { margin:.4rem 0 .6rem; font-size:clamp(2rem,5vw,3.35rem); }
        .hero p { color:var(--muted); font-size:1.03rem; max-width:780px; margin:0; }
        .prototype-notice { border-left:4px solid var(--gold); background:#fff8e8; color:#5f4a1e;
                            padding:.9rem 1rem; border-radius:0 12px 12px 0; margin:1rem 0 1.5rem; }
        .result-card { margin-top:.5rem; padding:1.25rem 1.35rem; border-radius:18px; background:var(--paper);
                       border:1px solid rgba(19,35,58,.11); box-shadow:0 10px 30px rgba(31,55,80,.06); }
        .result-card h3 { margin:.25rem 0 .8rem; }
        .prediction-line { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; }
        .prediction-label { display:inline-flex; padding:.36rem .72rem; border-radius:999px; font-weight:800; }
        .biased-label { color:#862f42; background:var(--rose-soft); }
        .neutral-label { color:#1f6284; background:var(--blue-soft); }
        .confidence-text { color:var(--muted); font-size:.92rem; }
        .probability-row { margin:.9rem .15rem; }
        .probability-label { display:flex; justify-content:space-between; color:var(--ink); margin-bottom:.35rem; }
        .probability-track { height:10px; overflow:hidden; background:#e6ebef; border-radius:999px; }
        .probability-fill { height:100%; border-radius:999px; }
        .neutral-fill { background:linear-gradient(90deg,#63a9ca,#2e6f95); }
        .biased-fill { background:linear-gradient(90deg,#d87a8c,#a8495b); }
        .flow { display:grid; grid-template-columns:repeat(5,1fr); gap:.6rem; align-items:center; margin:1.2rem 0 1.8rem; }
        .flow-step { min-height:88px; display:flex; flex-direction:column; align-items:center; justify-content:center;
                     text-align:center; padding:.75rem; background:#fff; border:1px solid rgba(19,35,58,.11);
                     border-radius:14px; color:var(--ink); font-weight:700; }
        .flow-step small { display:block; color:var(--muted); font-weight:500; margin-top:.25rem; }
        div[data-testid="stSidebar"] { background:#eef4f7; }
        @media (max-width:760px) { .flow { grid-template-columns:1fr; } .hero { padding:1.5rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title=APP_TITLE, page_icon="📰", layout="wide")
inject_styles()

if "current_sentence" not in st.session_state:
    st.session_state.current_sentence = SAMPLE_SENTENCES["Potentially biased example"]
if "prediction" not in st.session_state:
    st.session_state.prediction = None

with st.sidebar:
    st.markdown("## Prototype controls")
    st.info("Baseline interface only")
    st.markdown("---")
    st.markdown("**Prototype status**")
    st.success("Real trained baseline enabled")
    st.caption("Checkpoint: BABE Fold 1, initialized from the corrected WNC baseline.")
    st.caption("The enhanced model is reserved for Thesis 2 and is not displayed.")

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">SENTENCE-LEVEL MEDIA BIAS</div>
        <h1>How a news sentence becomes a prediction</h1>
        <p>A working prototype using a genuinely trained DA-RoBERTa baseline checkpoint.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="prototype-notice">
        <strong>Prototype scope:</strong> Predictions come from the real one-fold baseline checkpoint.
        They demonstrate the interface and inference process, but should not be treated as factual
        judgments or as a replacement for the thesis's five-fold experimental evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)

analyzer_tab, process_tab, study_tab = st.tabs(
    ["Analyze sentence", "How the system works", "About the study"]
)

with analyzer_tab:
    st.subheader("Enter one English news sentence")
    sample_choice = st.selectbox("Load an example", ["Keep current text", *SAMPLE_SENTENCES.keys()])
    if sample_choice != "Keep current text":
        st.session_state.current_sentence = SAMPLE_SENTENCES[sample_choice]

    sentence = st.text_area(
        "News sentence",
        key="current_sentence",
        height=145,
        max_chars=1500,
        placeholder="Paste or type one English news sentence here...",
    )

    if st.button("Analyze with the trained baseline", type="primary", use_container_width=True):
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            st.warning("Enter a sentence before generating a prediction.")
            st.session_state.prediction = None
        else:
            try:
                with st.spinner("Running real DA-RoBERTa inference..."):
                    st.session_state.prediction = predict_with_baseline(cleaned_sentence)
            except Exception as error:
                st.session_state.prediction = None
                st.error("The trained model could not be loaded or executed.")
                st.exception(error)

    if st.session_state.prediction is not None:
        prediction = st.session_state.prediction
        st.markdown("### Baseline output")
        result_card(prediction)
        st.download_button(
            "Download this real baseline output as JSON",
            data=prediction_payload(sentence, prediction),
            file_name="da_roberta_baseline_output.json",
            mime="application/json",
            use_container_width=True,
        )

with process_tab:
    st.subheader("Actual inference pipeline")
    st.markdown(
        """
        <div class="flow">
            <div class="flow-step">1. Sentence input<small>One English news sentence</small></div>
            <div class="flow-step">2. RoBERTa tokenization<small>Input IDs and attention mask</small></div>
            <div class="flow-step">3. RoBERTa encoder<small>Contextual representation</small></div>
            <div class="flow-step">4. Classification head<small>Two real class logits</small></div>
            <div class="flow-step">5. Decision<small>Softmax probabilities and label</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        The application loads the Fold 1 BABE checkpoint initialized from the corrected WNC
        baseline. The actual RoBERTa tokenizer converts a submitted sentence into input IDs
        and an attention mask. The encoder and classification head produce logits for
        **non-biased** and **biased**, and softmax converts those logits into probabilities.

        The enhanced pairwise-ranking model is planned for Thesis 2 and is intentionally
        excluded until it has been trained and evaluated.
        """
    )

with study_tab:
    st.subheader("Prototype boundaries")
    st.markdown(
        """
        **Included in this prototype**

        - Real DA-RoBERTa baseline inference
        - A BABE Fold 1 checkpoint initialized from the corrected WNC baseline
        - Actual RoBERTa tokens, transformed values, logits, probabilities, and label
        - Downloadable prediction details

        **Not included in this prototype**

        - The proposed enhanced pairwise-ranking model
        - Five-model ensembling
        - A claim that the prototype prediction is an objective factual judgment

        The thesis's five-fold results remain the experimental evaluation. This single-fold
        checkpoint is used specifically to demonstrate deployable baseline inference.
        """
    )
