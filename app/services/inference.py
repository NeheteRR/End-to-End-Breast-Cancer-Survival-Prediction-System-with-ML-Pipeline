"""
app/services/inference.py

Patient-level prediction logic.
Accepts a plain dict of clinical values, builds the correctly-shaped
feature row, and returns prediction + probability.
"""

import pandas as pd
import numpy as np

from app.services.model_io import load_model, load_feature_columns


def build_patient_row(patient_dict: dict, feature_cols: list[str]) -> pd.DataFrame:
    """
    Convert a flat dict of patient values into a one-row DataFrame that
    matches the training feature schema (all columns present, zeros by default).

    Unknown keys in patient_dict are silently ignored with a warning.
    """
    row = pd.DataFrame(columns=feature_cols)
    row.loc[0] = 0.0  # initialise every encoded column to 0

    for key, val in patient_dict.items():
        if key in feature_cols:
            row.loc[0, key] = val
        else:
            print(f"  ⚠  Feature '{key}' not found in model schema — skipped.")

    return row


def predict(patient_dict: dict, model=None, feature_cols: list[str] = None) -> dict:
    """
    Make a prediction for a single patient.

    Args:
        patient_dict  — raw clinical values keyed by (possibly one-hot) feature name
        model         — fitted estimator (loaded from disk if None)
        feature_cols  — ordered feature list (loaded from disk if None)

    Returns:
        {
            "prediction":   int   (1 = Responder, 0 = Non-Responder),
            "probability":  float (probability of being a responder),
            "patient_row":  pd.DataFrame
        }
    """
    if model is None:
        model = load_model()
    if feature_cols is None:
        feature_cols = load_feature_columns()

    patient_row = build_patient_row(patient_dict, feature_cols)
    prediction  = int(model.predict(patient_row)[0])
    probability = float(model.predict_proba(patient_row)[0][1])

    return {
        "prediction":  prediction,
        "probability": round(probability, 4),
        "patient_row": patient_row,
    }
