"""
tabular_predict.py
==================
Inference module for the trained tabular disease classifier.
Import this in app.py — do NOT call fit() here.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

MODEL_DIR      = "models"
PIPELINE_PATH  = os.path.join(MODEL_DIR, "tabular_pipeline.pkl")
META_PATH      = os.path.join(MODEL_DIR, "tabular_meta.json")

_pipeline = None
_meta     = None


def _load():
    global _pipeline, _meta
    if _pipeline is None:
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                f"Tabular model not found at '{PIPELINE_PATH}'. "
                "Run: python3 train_tabular_model.py"
            )
        _pipeline = joblib.load(PIPELINE_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _pipeline, _meta


def get_feature_schema():
    """Return list of required feature column names."""
    _, meta = _load()
    return meta["feature_columns"]


def predict_disease(input_dict: dict) -> dict:
    """
    Predict disease category from a patient feature dict.

    Parameters
    ----------
    input_dict : dict
        Keys = feature column names (matching training schema).
        Missing values should be None or omitted.

    Returns
    -------
    dict with keys:
        predicted_class  : str
        class_scores     : {class_name: probability}
        disclaimer       : str
    """
    pipeline, meta = _load()
    feature_cols = meta["feature_columns"]
    class_names  = meta["class_names"]

    # Build a single-row DataFrame with correct columns
    row = {col: input_dict.get(col, np.nan) for col in feature_cols}
    X = pd.DataFrame([row])

    # Convert numeric columns properly (strings → float)
    for col in meta["numeric_columns"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    proba = pipeline.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = pipeline.classes_[pred_idx]

    scores = {cls: round(float(p), 4)
              for cls, p in zip(pipeline.classes_, proba)}

    return {
        "predicted_class": pred_class,
        "class_scores": scores,
        "disclaimer": meta["disclaimer"],
    }
