"""
Train five classifiers on the UCI Breast Cancer Wisconsin (Diagnostic) dataset,
evaluate them on a held-out test split, and save everything the Streamlit app needs.

Run:  python train_models.py

Outputs
  model/<name>.joblib    one fitted pipeline per classifier (preprocessing included)
  model/metrics.csv      evaluation metrics of every model on the test split
  model/feature_names.json   input columns the models expect, in order
  test_data.csv          the held-out test split (features + target), used by the app
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "target"            # 1 = malignant, 0 = benign
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

# Display name -> file name. The app reads the same mapping from metrics.csv.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_dataset() -> pd.DataFrame:
    """UCI Breast Cancer Wisconsin (Diagnostic), shipped with scikit-learn.

    scikit-learn encodes 0 = malignant, 1 = benign. We flip it so that the
    positive class (1) is malignant, which is the case we care about catching.
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df[TARGET] = 1 - df["target"]
    return df


def build_models() -> dict:
    # Scale-sensitive models get a StandardScaler inside their pipeline, so the
    # scaler fitted on the training data travels with the saved model.
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            # SVC has no native probabilities; Platt scaling via CalibratedClassifierCV
            # gives the scores AUC needs (the recommended replacement for probability=True)
            ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE),
                                           method="sigmoid", ensemble=False, cv=5)),
        ]),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main() -> None:
    df = load_dataset()
    features = [c for c in df.columns if c != TARGET]
    X, y = df[features], df[TARGET]

    print(f"Dataset: {df.shape[0]} instances, {len(features)} features")
    print(f"Class balance: malignant={int(y.sum())}, benign={int((1 - y).sum())}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
        scores = evaluate(model, X_test, y_test)
        rows.append({"Model": name, **scores})

    metrics = pd.DataFrame(rows).set_index("Model").round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv")

    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump({"features": features, "target": TARGET,
                   "model_files": MODEL_FILES}, f, indent=2)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    pd.set_option("display.width", 120)
    print("Evaluation on the held-out test set")
    print(metrics.to_string())
    print(f"\nSaved {len(MODEL_FILES)} models to {MODEL_DIR}/ and test split to test_data.csv")


if __name__ == "__main__":
    main()
