# DA-RoBERTa Streamlit Prototype

This prototype performs real sentence-level inference with a trained DA-RoBERTa
baseline checkpoint. The checkpoint was initialized from the corrected WNC model
and fine-tuned using BABE Fold 1. The thesis's existing five-fold results remain the
experimental evaluation; this single-fold checkpoint is for deployment demonstration.

## Run locally

From this folder, install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put this folder in a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Select **Create app** and choose the repository and branch.
4. Set the main file path to `streamlit_prototype/app.py` if the repository contains
   the whole thesis workspace, or `app.py` if this folder is the repository root.
5. Deploy the app. The public checkpoint bundle is downloaded from
   `clarkkyvii/da-roberta-baseline-prototype` on Hugging Face and cached when the
   first inference is requested. No Streamlit secret or GPU is required.

## Prototype scope

The interface displays the actual RoBERTa tokens, transformed representation preview,
class logits, softmax probabilities, and predicted label. It does not include or
simulate the proposed enhanced pairwise-ranking model, which is reserved for Thesis 2.
