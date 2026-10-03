"""Train the loan-status demo models for deployment.

This keeps the original project setup so the existing KNN (k=3) result remains
comparable: dropna -> drop applicant_id -> one-hot encoding -> 67/33 split
(random_state=42) -> SMOTE on training data only -> Naive Bayes + KNN.

Extra deployment metadata is saved for the Streamlit app:
- confusion matrices
- per-class classification reports
- class counts
- package versions
- input ranges/categories

Usage:
    python train_model_deploy.py "loan_data (1) (1).csv"
"""

import sys

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "loan_data (1) (1).csv"
TARGET = "loan_status"
ID_COL = "applicant_id"
OUT_PATH = "loan_model_deploy.joblib"
KNN_K = 3

# -------------------- Load data --------------------
df_raw = pd.read_csv(CSV_PATH)
missing_before = df_raw.isna().sum().sum()
original_rows = len(df_raw)

df = df_raw.dropna().copy()
rows_after_dropna = len(df)
df = df.drop(columns=[ID_COL], errors="ignore")

X = df.drop(columns=TARGET)
le = LabelEncoder()
y = le.fit_transform(df[TARGET])

cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
num_cols = [c for c in X.columns if c not in cat_cols]


def make_prep():
    return ColumnTransformer(
        [("cat", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
            dtype=int,
        ), cat_cols)],
        remainder="passthrough",
    ).set_output(transform="pandas")


pipelines = {
    "Naïve Bayes": Pipeline([
        ("prep", make_prep()),
        ("smote", SMOTE(random_state=42)),
        ("clf", GaussianNB()),
    ]),
    f"KNN (k={KNN_K})": Pipeline([
        ("prep", make_prep()),
        ("smote", SMOTE(random_state=42)),
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=KNN_K)),
    ]),
}

# Keep the original project's 67/33 split so metrics remain comparable.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

models = {}
labels = list(range(len(le.classes_)))
label_names = [str(x) for x in le.classes_]

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n=== {name} ===")
    print(classification_report(
        y_test, y_pred, labels=labels,
        target_names=label_names, zero_division=0
    ))
    print("Confusion matrix:")
    print(cm)

    models[name] = {
        "pipeline": pipe,
        "metrics": {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_size": int(len(y_test)),
    }

# -------------------- Input metadata --------------------
numeric_meta = {}
for c in num_cols:
    col = X[c]
    numeric_meta[c] = {
        "min": float(col.min()),
        "max": float(col.max()),
        "median": float(col.median()),
        "is_int": bool(np.all(np.mod(col, 1) == 0)),
        "is_binary": bool(set(col.dropna().unique()) <= {0, 1}),
    }

categorical_meta = {
    c: sorted(X[c].astype(str).unique().tolist()) for c in cat_cols
}

class_counts = {
    str(label): int((df[TARGET].astype(str) == str(label)).sum())
    for label in le.classes_
}

bundle = {
    "models": models,
    "classes": le.classes_.astype(str).tolist(),
    "feature_order": X.columns.tolist(),
    "numeric": numeric_meta,
    "categorical": categorical_meta,
    "n_rows": int(len(df)),
    "data_summary": {
        "original_rows": int(original_rows),
        "rows_after_dropna": int(rows_after_dropna),
        "missing_values_before_dropna": int(missing_before),
        "class_counts": class_counts,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    },
    "training": {
        "target": TARGET,
        "split": "67/33",
        "random_state": 42,
        "smote_random_state": 42,
        "knn_k": KNN_K,
    },
    "versions": {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
    "purpose": "Educational/demo loan-status classification. Not for automated credit decisions.",
}

joblib.dump(bundle, OUT_PATH)
print(f"\nSaved {OUT_PATH}")
