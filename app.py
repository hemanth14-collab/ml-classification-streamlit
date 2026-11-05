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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]
METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
ERROR_COLS = ["Missed malignant (FN)", "False alarms (FP)"]

# Colours match .streamlit/config.toml so the charts and the UI look like one app
TEAL = "#0F766E"
TEAL_CMAP = LinearSegmentedColormap.from_list("teal", ["#F0FDFA", TEAL])
MODEL_COLORS = {
    "Logistic Regression": "#0F766E",
    "Decision Tree": "#B45309",
    "KNN": "#6D28D9",
    "Naive Bayes": "#BE185D",
    "Support Vector Machine": "#1D4ED8",
}

# Short plain-English explanations shown as tooltips next to each metric
METRIC_HELP = {
    "Accuracy": "Share of all cases classified correctly.",
    "AUC": "How well the malignancy scores rank malignant cases above benign ones (1.0 = perfect ranking).",
    "Precision": "Of the cases flagged malignant, the share that really were malignant.",
    "Recall": "Of the truly malignant cases, the share the model caught. The key number for screening.",
    "F1": "Harmonic mean of Precision and Recall.",
    "MCC": "Matthews correlation, from -1 to 1. Stays fair when one class is much larger than the other.",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison", page_icon="🩺", layout="wide")


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
    (_, fp), (fn, _) = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
        "Missed malignant (FN)": int(fn),
        "False alarms (FP)": int(fp),
    }


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    # Name each cell so the matrix can be read without remembering the layout
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    labels = np.char.add(np.char.add(cm.astype(str), "\n"), cell_names)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=labels, fmt="", cmap=TEAL_CMAP, cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=1, linecolor="white", annot_kws={"size": 13})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_metric_bars(table: pd.DataFrame):
    """Grouped bars: one group per metric, one coloured bar per model."""
    fig, ax = plt.subplots(figsize=(10, 4))
    table[METRIC_COLS].T.plot(kind="bar", ax=ax, width=0.8,
                              color=[MODEL_COLORS.get(n, "#64748B") for n in table.index])
    ax.set_ylim(max(0.0, np.nanmin(table[METRIC_COLS].values) - 0.05), 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_roc_curves(y, results: dict, highlight: str):
    """All five ROC curves on one chart; the sidebar's model is drawn thicker."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["y_prob"])
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, "#64748B"),
                lw=3 if name == highlight else 1.4,
                alpha=1.0 if name == highlight else 0.75,
                label=f"{name} (AUC {res['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#94A3B8", label="Random guess")
    ax.set_xlabel("False positive rate (benign flagged as malignant)")
    ax.set_ylabel("True positive rate (malignant caught)")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
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
st.title("🩺 Breast Cancer Diagnosis: Classifier Comparison")
st.markdown(
    "Five classifiers trained on the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset "
    "(30 features, 569 instances). Upload test data to see how each saved model performs. "
    "Models are loaded from disk and only used for prediction; nothing is retrained."
)
st.info(
    "**Reading the results:** in cancer screening a missed malignant tumour (false negative) "
    "is far worse than a false alarm, so watch **Recall** and the **Missed malignant** count, "
    "not just Accuracy. Hover over any metric for a short explanation.",
    icon="💡",
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

tab_single, tab_compare, tab_roc, tab_case = st.tabs(
    [f"Selected model: {model_name}", "Compare all models", "ROC curves", "Single-case explorer"])

with tab_single:
    r = results[model_name]
    st.subheader(f"Evaluation metrics: {model_name}")
    cols = st.columns(len(METRIC_COLS))
    for col, metric in zip(cols, METRIC_COLS):
        col.metric(metric, "n/a" if np.isnan(r[metric]) else f"{r[metric]:.4f}",
                   help=METRIC_HELP[metric])
    st.caption(f"Errors: **{r['Missed malignant (FN)']}** malignant case(s) missed, "
               f"**{r['False alarms (FP)']}** benign case(s) wrongly flagged.")

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
    table = pd.DataFrame({n: {m: res[m] for m in METRIC_COLS + ERROR_COLS}
                          for n, res in results.items()}).T
    table[ERROR_COLS] = table[ERROR_COLS].astype(int)
    table.index.name = "ML Model Name"
    # Highlight the best model in each column: highest score, fewest errors
    styled = (table.style
              .format("{:.4f}", subset=METRIC_COLS)
              .highlight_max(subset=METRIC_COLS, color="#CCFBF1")
              .highlight_min(subset=ERROR_COLS, color="#CCFBF1"))
    st.dataframe(styled)
    st.caption("Shaded cells mark the best model for each column.")

    st.pyplot(plot_metric_bars(table))

    best = table["MCC"].idxmax()
    st.success(f"Best model on this data by MCC: **{best}** "
               f"(MCC {table.loc[best, 'MCC']:.4f}, F1 {table.loc[best, 'F1']:.4f})")

with tab_roc:
    st.subheader("ROC curves for all five models")
    if y.nunique() < 2:
        st.info("ROC curves need both benign and malignant cases in the data.")
    else:
        chart_col, note_col = st.columns([1.4, 1])
        with chart_col:
            st.pyplot(plot_roc_curves(y, results, highlight=model_name))
        with note_col:
            st.markdown(
                "Each curve shows the trade-off as the decision threshold moves: "
                "going right catches more malignant cases but raises more false alarms. "
                "A curve that hugs the top-left corner is better, and the area under it is the **AUC**.\n\n"
                f"The model picked in the sidebar (**{model_name}**) is drawn thicker. "
                "Notice how the Decision Tree's curve is made of a few straight segments: "
                "a single tree outputs only a handful of distinct probabilities."
            )

with tab_case:
    st.subheader("What does each model say about one case?")
    # Cases where at least one model disagrees with the others are the interesting ones
    all_preds = np.column_stack([results[n]["y_pred"] for n in models])
    disputed = np.flatnonzero(all_preds.min(axis=1) != all_preds.max(axis=1))

    only_disputed = st.toggle(f"Only show the {len(disputed)} case(s) where the models disagree",
                              value=len(disputed) > 0, disabled=len(disputed) == 0,
                              key="case_only_disputed")
    positions = disputed.tolist() if only_disputed else list(range(len(df)))
    pos = st.selectbox(
        "Pick a case (row number in the test data)", positions,
        format_func=lambda i: f"Row {i + 1}  ·  actual: {CLASS_NAMES[int(y.iloc[i])]}",
        key="case_pick",
    )

    actual = int(y.iloc[pos])
    case_rows = []
    for name, res in results.items():
        pred = int(res["y_pred"][pos])
        case_rows.append({
            "Model": name,
            "Prediction": CLASS_NAMES[pred],
            "P(malignant)": float(res["y_prob"][pos]),
            "Correct?": "✅" if pred == actual else "❌",
        })
    case_table = pd.DataFrame(case_rows).set_index("Model")
    votes = int((all_preds[pos] == 1).sum())

    left, right = st.columns([1.6, 1])
    with left:
        st.dataframe(
            case_table,
            column_config={"P(malignant)": st.column_config.ProgressColumn(
                "P(malignant)", min_value=0.0, max_value=1.0, format="%.3f")},
        )
    with right:
        st.metric("Actual diagnosis", CLASS_NAMES[actual])
        st.metric("Models voting malignant", f"{votes} of {len(models)}")
    with st.expander("Feature values for this case"):
        st.dataframe(X.iloc[[pos]].T.set_axis(["value"], axis=1))
