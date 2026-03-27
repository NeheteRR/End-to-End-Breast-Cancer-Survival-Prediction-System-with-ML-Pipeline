import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def get_explainer(model, X_background):
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_background)
    return shap.TreeExplainer(model)


def explain_patient(patient_row, model, feature_cols, explainer):
    pred      = int(model.predict(patient_row)[0])
    prob      = float(model.predict_proba(patient_row)[0][1])
    shap_vals = explainer.shap_values(patient_row)

    # TreeExplainer (binary RF) → list; LinearExplainer → 2D array
    if isinstance(shap_vals, list):
        local_vals = shap_vals[1][0]
    else:
        local_vals = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

    shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "value":      patient_row.iloc[0].values,
        "shap_value": local_vals,
    }).sort_values("shap_value", ascending=False)

    top_pos = shap_df.head(5)
    top_neg = shap_df.tail(5)

    text_report = (
        f"\nPredicted: {'Responder' if pred == 1 else 'Non-Responder'}  "
        f"(probability: {prob:.3f})\n\n"
        "Top features INCREASING response probability:\n"
        f"{top_pos[['feature', 'shap_value']].to_string(index=False)}\n\n"
        "Top features DECREASING response probability:\n"
        f"{top_neg[['feature', 'shap_value']].to_string(index=False)}\n"
    )

    return {
        "prediction":   pred,
        "probability":  prob,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "text_report":  text_report,
        "shap_df":      shap_df,
    }


def save_summary_plot(shap_values, X, output_path, plot_type="bar"):
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    plt.figure()
    shap.summary_plot(vals, X, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
