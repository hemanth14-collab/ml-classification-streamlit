# Breast Cancer Diagnosis: Comparing Five Classification Models

An end-to-end ML project: five classifiers are trained on the same dataset, evaluated with six metrics, saved to disk, and served through an interactive Streamlit app where anyone can upload test data and see how each model performs.

## a. Problem statement

Given 30 numeric measurements computed from a digitised image of a fine needle aspirate (FNA) of a breast mass, predict whether the tumour is **malignant (1)** or **benign (0)**.

This is a binary classification problem. Missing a malignant case (a false negative) is far more costly than a false alarm, so along with accuracy the comparison pays close attention to **Recall** and **MCC**, which stay honest when the classes are imbalanced.

The task is to implement Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes and Support Vector Machine on the same data, compare them on Accuracy, AUC, Precision, Recall, F1 and MCC, and deploy the trained models in a Streamlit web app.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository, dataset id 17 ([link](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)); loaded via `sklearn.datasets.load_breast_cancer`, which ships the same data |
| Instances | 569 |
| Features | 30, all continuous (no missing values) |
| Target | `target`: 1 = malignant (212 rows, 37%), 0 = benign (357 rows, 63%) |
| Train / test split | 80 / 20, stratified, `random_state=42` → 455 train, 114 test |
| Test file | `test_data.csv` (114 rows: 30 features + `target`; 42 malignant, 72 benign) |

The 30 features are 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each recorded three ways: the **mean**, the **standard error**, and the **worst** (mean of the three largest values) across the nuclei in the image.

Two things about the data shape the results: the feature scales differ hugely (`mean area` runs from 143 to 2,501 while `smoothness error` sits around 0.002 to 0.03), and many features are near-duplicates (`mean radius`, `mean perimeter` and `mean area` correlate at 0.99). Logistic Regression, KNN and SVM are therefore trained inside a pipeline with `StandardScaler`, fitted on the training data only and saved together with each model.

## c. GitHub repository link

**Repository:** https://github.com/hemanth14-collab/ml-classification-streamlit

**Live Streamlit app:** https://ml-classificatio-ehspjndqyzcajw7fsfxlhd.streamlit.app

```
ml-classification-streamlit/
├── app.py                 # Streamlit app (loads saved models, never retrains)
├── train_models.py        # trains all 5 models, computes metrics, saves artifacts
├── requirements.txt
├── README.md
├── test_data.csv          # held-out test split used in the experiments
├── .streamlit/
│   └── config.toml        # app theme
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── feature_names.json # feature order, target name, model file map
    └── metrics.csv        # test-set metrics written by train_models.py
```

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` → `LogisticRegression(max_iter=5000)` |
| Decision Tree | `DecisionTreeClassifier(random_state=42)`, unpruned (grows to depth 8, 24 leaves) |
| KNN | `StandardScaler` → `KNeighborsClassifier(n_neighbors=5)` |
| Naive Bayes | `GaussianNB()` (features are continuous, so Gaussian rather than Multinomial) |
| Support Vector Machine | `StandardScaler` → `SVC(kernel="rbf")` wrapped in `CalibratedClassifierCV(method="sigmoid", cv=5)` for probability scores (needed for AUC) |

### Comparison table (held-out test set, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | 0.9756 | **0.9524** | **0.9639** | **0.9433** |

Positive class = malignant (1). Precision, Recall and F1 are for the malignant class.

Errors behind those numbers (out of 42 malignant and 72 benign test cases):

| Model | Missed malignant (FN) | False alarms (FP) |
|---|---|---|
| Logistic Regression | 3 | 1 |
| Decision Tree | 4 | 4 |
| KNN | 4 | 1 |
| Naive Bayes | 7 | 0 |
| Support Vector Machine | 2 | 1 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for such a simple model: 96.5% accuracy and the best AUC (0.996), meaning its probability ranking of malignant vs benign is almost perfect. It missed 3 malignant cases and raised 1 false alarm. The classes are close to linearly separable once features are scaled, which is why a linear boundary does this well. It was also the most stable model in 5-fold cross-validation (MCC 0.945 ± 0.034). |
| Decision Tree | Weakest model on every metric (accuracy 0.930, MCC 0.849) and the only one with clearly lower AUC (0.925), since a single tree gives coarse, near 0/1 probabilities. Left unpruned, it grew to depth 8 with 24 leaves and overfits the 455 training rows: 4 missed malignant cases plus 4 false alarms. It also had the highest variance across CV folds (± 0.061). Limiting depth or using an ensemble such as Random Forest would help. |
| KNN | Solid with scaling (accuracy 0.956, MCC 0.906, precision 0.974), but it missed 4 malignant cases, so recall (0.905) trails the top models. Distance-based voting suffers somewhat in 30 dimensions where many features are redundant, and it must keep the whole training set in memory for prediction. |
| Naive Bayes | Perfect precision (1.000): every case it flagged as malignant really was malignant. But it was the most cautious model, missing 7 of 42 malignant cases (recall 0.833), which is the worst outcome for a screening task. The independence assumption is badly violated here (radius, perimeter and area correlate at 0.99), so the same evidence gets counted several times and skews the decisions. Its AUC is still high (0.993), so lowering the decision threshold would recover much of the lost recall. |
| Support Vector Machine | Best overall on the test set: top accuracy (0.974), recall (0.952), F1 (0.964) and MCC (0.943), missing only 2 of 42 malignant cases with 1 false alarm. The RBF kernel captures the slight non-linearity in the boundary that a linear model misses, and maximising the margin keeps it robust on unseen data. |
| **Overall winner for this dataset** | **Support Vector Machine.** It has the highest MCC and F1 and, more importantly for a cancer screening task, the fewest missed malignant cases. Logistic Regression is a very close second (best AUC, and marginally ahead in 5-fold cross-validation at 0.945 vs 0.930 MCC), so on a different split the two could swap places; both clearly outperform the other three models. If interpretability mattered more than the last bit of recall, Logistic Regression would be the practical choice. |

## Streamlit app features

- **CSV upload** for test data (a sample `test_data.csv` is bundled and also downloadable from the sidebar)
- **Model selection dropdown** with all five classifiers
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- **Confusion matrix and classification report** for the selected model, with TN/FP/FN/TP labelled on each cell
- **Metric tooltips** that explain each score in plain English, plus a count of missed malignant cases and false alarms
- **Download predictions**: row-by-row predictions and malignancy probabilities for the selected model as a CSV
- **Compare all models** tab: metrics table (best model per column shaded, with missed-malignant and false-alarm counts) and a bar chart grouped by metric
- **ROC curves** tab: all five ROC curves on one chart, with the selected model highlighted
- **Single-case explorer** tab: pick one row and see every model's prediction and malignancy probability side by side; by default it lists only the cases where the models disagree
- **Custom theme** (`.streamlit/config.toml`): teal clinical palette used by both the UI and the charts

The app loads the saved `.joblib` pipelines from `model/` and only predicts on the uploaded file; it never retrains.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py     # optional: retrains and regenerates model/ and test_data.csv
streamlit run app.py
```

The uploaded CSV must contain all 30 feature columns (same names as `test_data.csv`) plus a `target` column with values 0 or 1.

## Deployment

Deployed on Streamlit Community Cloud from the `main` branch with `app.py` as the entry point and **Python 3.13** selected under Advanced settings. The saved models were trained with scikit-learn 1.9.1, which `requirements.txt` pins so that they load correctly.
