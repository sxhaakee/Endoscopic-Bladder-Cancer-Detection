"""
train_tabular_model.py
======================
Reproducible training pipeline for the 5-class disease classification
using 'bladder data.csv'.

Classes: UrinaryBladder, Prostate, Kidney, Uterus, Cystitis

Run:
    python3 train_tabular_model.py

Outputs (saved to models/ folder):
    tabular_pipeline.pkl       — full fitted sklearn Pipeline
    tabular_meta.json          — feature schema, class names, evaluation
    tabular_eval.json          — detailed metrics
    confusion_matrix.png       — confusion matrix image
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "bladder data.csv"
OUTPUT_DIR  = "models"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

TARGET_COL  = "Disease"
ID_COL      = "Patient Number"   # identifier — excluded from features

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  TABULAR ML TRAINING PIPELINE — BladderVision AI")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Basic checks ───────────────────────────────────────────────────────────
print(f"\nTarget class distribution:")
print(df[TARGET_COL].value_counts().to_string())

dups = df.duplicated().sum()
print(f"\nDuplicate rows: {dups}")
if dups > 0:
    df = df.drop_duplicates()
    print(f"  → Removed duplicates. Remaining: {len(df)}")

# Drop missing targets
before = len(df)
df = df.dropna(subset=[TARGET_COL])
if len(df) < before:
    print(f"  → Dropped {before - len(df)} rows with missing Disease label.")

# ── 3. Feature / target split ─────────────────────────────────────────────────
drop_cols = [ID_COL, TARGET_COL]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
print(f"Classes: {sorted(y.unique())}")

# ── 4. Identify numeric and categorical features ───────────────────────────────
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()
print(f"\nNumeric cols ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

# ── 5. Preprocessing pipelines ───────────────────────────────────────────────
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
], remainder="drop")

# ── 6. Train/test split (stratified) ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"\nTrain set: {len(X_train)}  |  Test set: {len(X_test)}")

# ── 7. Build full pipeline ────────────────────────────────────────────────────
rf_pipeline = Pipeline([
    ("prep",       preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",   # handles class imbalance
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )),
])

# ── 8. Cross-validation on training data ─────────────────────────────────────
print("\nRunning 5-fold stratified cross-validation on training data …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_results = cross_validate(
    rf_pipeline, X_train, y_train, cv=cv,
    scoring=["accuracy", "f1_macro"],
    return_train_score=False,
)
cv_acc   = cv_results["test_accuracy"].mean()
cv_acc_s = cv_results["test_accuracy"].std()
cv_f1    = cv_results["test_f1_macro"].mean()
cv_f1_s  = cv_results["test_f1_macro"].std()
print(f"  CV Accuracy:  {cv_acc:.4f} ± {cv_acc_s:.4f}")
print(f"  CV F1-macro:  {cv_f1:.4f} ± {cv_f1_s:.4f}")

# ── 9. Final fit on full training set ────────────────────────────────────────
print("\nFitting final pipeline on full training set …")
rf_pipeline.fit(X_train, y_train)

# ── 10. Evaluate on held-out test set ────────────────────────────────────────
y_pred = rf_pipeline.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
f1m  = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, output_dict=True)
report_str = classification_report(y_test, y_pred)

print("\n--- TEST SET RESULTS ---")
print(f"Accuracy : {acc:.4f}")
print(f"F1-macro : {f1m:.4f}")
print("\nClassification Report:")
print(report_str)

# ── 11. Confusion matrix plot ─────────────────────────────────────────────────
class_names = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Tabular Disease Classifier (Test Set)")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"\nConfusion matrix saved: {cm_path}")

# ── 12. Save pipeline ─────────────────────────────────────────────────────────
pipeline_path = os.path.join(OUTPUT_DIR, "tabular_pipeline.pkl")
joblib.dump(rf_pipeline, pipeline_path)
print(f"Pipeline saved: {pipeline_path}")

# ── 13. Save metadata & evaluation JSON ───────────────────────────────────────
meta = {
    "feature_columns":  feature_cols,
    "numeric_columns":  num_cols,
    "categorical_columns": cat_cols,
    "target_column":    TARGET_COL,
    "id_column_excluded": ID_COL,
    "class_names":      class_names,
    "random_seed":      RANDOM_SEED,
    "test_size":        TEST_SIZE,
    "train_rows":       int(len(X_train)),
    "test_rows":        int(len(X_test)),
    "model":            "RandomForestClassifier(n_estimators=200, class_weight=balanced)",
    "cv_accuracy_mean": round(float(cv_acc), 4),
    "cv_accuracy_std":  round(float(cv_acc_s), 4),
    "cv_f1_macro_mean": round(float(cv_f1), 4),
    "cv_f1_macro_std":  round(float(cv_f1_s), 4),
    "test_accuracy":    round(float(acc), 4),
    "test_f1_macro":    round(float(f1m), 4),
    "pipeline_path":    pipeline_path,
    "confusion_matrix_path": cm_path,
    "disclaimer": (
        "This model is experimental and intended for a college project only. "
        "It is NOT a certified medical diagnostic tool."
    ),
}

meta_path = os.path.join(OUTPUT_DIR, "tabular_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved: {meta_path}")

eval_out = {
    "cv_results": {
        "accuracy_per_fold": cv_results["test_accuracy"].tolist(),
        "f1_macro_per_fold": cv_results["test_f1_macro"].tolist(),
    },
    "test_set": {
        "accuracy":  round(float(acc), 4),
        "f1_macro":  round(float(f1m), 4),
        "per_class": report,
    },
    "class_distribution_train": y_train.value_counts().to_dict(),
    "class_distribution_test":  y_test.value_counts().to_dict(),
}
eval_path = os.path.join(OUTPUT_DIR, "tabular_eval.json")
with open(eval_path, "w") as f:
    json.dump(eval_out, f, indent=2, default=str)
print(f"Evaluation saved: {eval_path}")

print("\n" + "=" * 60)
print("  Training complete. Model is ready.")
print("=" * 60)
