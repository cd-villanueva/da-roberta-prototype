# DA-RoBERTa Streamlit Prototype

This prototype demonstrates the intended user-interface flow for sentence-level
media bias detection. It deliberately uses deterministic seeded values instead of
trained-model inference. The displayed outputs must not be reported as experimental
results.

## Run locally

From this folder, install the dependency and start the app:

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
5. Deploy the app. No secrets, datasets, GPUs, or model files are required.

## Switching to real inference later

Replace `create_seeded_prediction()` in `app.py` with code that loads the finalized
tokenizer and model checkpoint and returns real logits. Preserve the prototype-mode
disclaimer until every visible output comes from the trained checkpoint.

