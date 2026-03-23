"""
app/services/explainability.py

SHAP-based model explainability.
Provides both:
  - Global feature importance (training set)
  - Local explanation for a single patient (text-only, no Jupyter dependency)
"""

import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server / script use
import matplotlib.pyplot as plt


def get_explainer(model) -> shap.TreeExplainer:
    """Initialise a SHAP TreeExplainer for a tree-based model."""
    return shap.TreeExplainer(model)


def compute_global_shap(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for an entire dataset."""
    return explainer.shap_values(X)


def get_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Aggregate per-feature mean |SHAP| importance into a ranked DataFrame.

    Works for both binary (2-class list) and multi-class outputs.
    """
    if isinstance(shap_values, list):
        # Binary RF returns a list [class0_array, class1_array]
        vals = shap_values[1]
    else:
        vals = shap_values

    importance = np.abs(vals).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_patient(
    patient_row: pd.DataFrame,
    model,
    feature_cols: list[str],
    explainer: shap.TreeExplainer = None,
) -> dict:
    """
    Generate a text-based local SHAP explanation for one patient.

    Returns a dict with:
      - prediction, probability
      - top_positive: 5 features pushing prediction UP
      - top_negative: 5 features pushing prediction DOWN
      - text_report:  human-readable summary string
    """
    if explainer is None:
        explainer = get_explainer(model)

    pred        = int(model.predict(patient_row)[0])
    prob        = float(model.predict_proba(patient_row)[0][1])
    shap_vals   = explainer.shap_values(patient_row)

    # Handle list format (Random Forest binary classification)
    if isinstance(shap_vals, list):
        local_vals = shap_vals[1][0]
    else:
        local_vals = shap_vals[0]

    shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "value":      patient_row.iloc[0].values,
        "shap_value": local_vals,
    }).sort_values("shap_value", ascending=False)

    top_pos = shap_df.head(5)
    top_neg = shap_df.tail(5)

    text_report = (
        "\n======================================\n"
        " CHEMOTHERAPY RESPONSE EXPLANATION\n"
        "======================================\n\n"
        f"Predicted Class : {pred}  (1 = Responder, 0 = Non-Responder)\n"
        f"Response Probability : {prob:.3f}\n\n"
        "TOP FEATURES INCREASING Response Probability\n"
        "---------------------------------------------\n"
        f"{top_pos[['feature', 'value', 'shap_value']].to_string(index=False)}\n\n"
        "TOP FEATURES DECREASING Response Probability\n"
        "---------------------------------------------\n"
        f"{top_neg[['feature', 'value', 'shap_value']].to_string(index=False)}\n"
    )

    return {
        "prediction":   pred,
        "probability":  prob,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "text_report":  text_report,
        "shap_df":      shap_df,
    }


def save_summary_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_path: str,
    plot_type: str = "bar",
) -> None:
    """Save a SHAP summary plot to disk (no display)."""
    # Unwrap list format for RandomForest
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(vals, X, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP {plot_type} plot saved → {output_path}")
