"""
app/utils/helpers.py

Miscellaneous helper utilities used across the project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def map_ui_inputs_to_features(
    age: float,
    tumor_size: float,
    tumor_stage: int,
    histologic_grade: int,
    er_status: str,
    her2_status: str,
    pam50_subtype: str,
    menopausal_state: str,
    oncotree_code: str,
    surgery_type: str,
) -> dict:
    """
    Convert human-friendly UI dropdown/slider values into the one-hot
    encoded feature keys expected by the trained model.

    Returns a flat dict compatible with inference.predict().
    """
    features: dict = {
        "Age at Diagnosis":        age,
        "Tumor Size":              tumor_size,
        "Tumor Stage":             tumor_stage,
        "Neoplasm Histologic Grade": histologic_grade,
    }

    # One-hot style binary flags
    if er_status == "Positive":
        features["ER Status_Positive"] = 1

    if her2_status == "Positive":
        features["HER2 Status_Positive"] = 1

    pam50_col = f"Pam50 + Claudin-low subtype_{pam50_subtype}"
    features[pam50_col] = 1

    if menopausal_state == "Pre":
        features["Inferred Menopausal State_Pre"] = 1

    features[f"Oncotree Code_{oncotree_code}"] = 1

    if surgery_type == "MASTECTOMY":
        features["Type of Breast Surgery_MASTECTOMY"] = 1

    return features


def format_shap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready version of a SHAP feature DataFrame."""
    return (
        df[["feature", "value", "shap_value"]]
        .rename(columns={
            "feature":    "Feature",
            "value":      "Patient Value",
            "shap_value": "SHAP Impact",
        })
        .reset_index(drop=True)
    )
