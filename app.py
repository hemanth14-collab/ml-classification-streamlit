"""
Streamlit app: evaluate five pre-trained classifiers on uploaded test data.

The models are trained once by train_models.py and saved in model/.
This app only loads them and predicts; it never retrains on the uploaded data.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]
METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "feature_names.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname)
              for name, fname in meta["model_files"].items()}
    return meta, models


def compute_metrics(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    # AUC is undefined if the uploaded file contains only one class
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) == 2 else float("nan")
    return {
        "y_pred": y_pred,
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


try:
    meta, models = load_artifacts()
except Exception as exc:  # most often a scikit-learn version mismatch
    st.error(f"Could not load the saved models from model/: {exc}. "
             "Check that requirements.txt versions are installed.")
    st.stop()

FEATURES, TARGET = meta["features"], meta["target"]

# ----------------------------------------------------------------- sidebar
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + target column)", type=["csv"])
use_sample = st.sidebar.checkbox("If nothing is uploaded, use the bundled test_data.csv", value=True)
with open(SAMPLE_CSV, "rb") as f:
    st.sidebar.download_button("Download sample test_data.csv", f.read(),
                               file_name="test_data.csv", mime="text/csv")

st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a classifier", list(models.keys()))

# ----------------------------------------------------------------- header
st.title("Breast Cancer Diagnosis: Classifier Comparison")
st.markdown(
    "Five classifiers trained on the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset "
    "(30 features, 569 instances). Upload test data to see how each saved model performs. "
    "Models are loaded from disk and only used for prediction; nothing is retrained."
)

# ----------------------------------------------------------------- data
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    df = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("Upload a CSV in the sidebar, or tick the box to use the bundled test data.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing {len(missing)} required column(s): "
             f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    st.stop()

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y = df[TARGET].astype(int)
if not set(y.unique()) <= {0, 1}:
    st.error(f"Target column '{TARGET}' must contain only 0 (benign) and 1 (malignant).")
    st.stop()

if y.nunique() < 2:
    st.warning("The uploaded data contains only one class, so AUC is undefined and "
               "Precision, Recall, F1 and MCC are not meaningful. Use a file with both classes.")

c1, c2, c3 = st.columns(3)
c1.metric("Rows evaluated", len(df))
c2.metric("Malignant (1)", int(y.sum()))
c3.metric("Benign (0)", int((y == 0).sum()))
st.caption(f"Data source: {source}")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20))

# ----------------------------------------------------------------- results
results = {name: compute_metrics(m, X, y) for name, m in models.items()}

tab_single, tab_compare = st.tabs([f"Selected model: {model_name}", "Compare all models"])

with tab_single:
    r = results[model_name]
    st.subheader(f"Evaluation metrics: {model_name}")
    cols = st.columns(len(METRIC_COLS))
    for col, metric in zip(cols, METRIC_COLS):
        col.metric(metric, "n/a" if np.isnan(r[metric]) else f"{r[metric]:.4f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(y, r["y_pred"], model_name))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, r["y_pred"], labels=[0, 1],
                                       target_names=CLASS_NAMES, output_dict=True,
                                       zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4))

with tab_compare:
    st.subheader("All five models on this test data")
    table = pd.DataFrame({n: {m: res[m] for m in METRIC_COLS} for n, res in results.items()}).T
    table.index.name = "ML Model Name"
    st.dataframe(table.round(4))

    fig, ax = plt.subplots(figsize=(10, 4))
    table.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(max(0.0, np.nanmin(table.values) - 0.05), 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    best = table["MCC"].idxmax()
    st.success(f"Best model on this data by MCC: **{best}** "
               f"(MCC {table.loc[best, 'MCC']:.4f}, F1 {table.loc[best, 'F1']:.4f})")
