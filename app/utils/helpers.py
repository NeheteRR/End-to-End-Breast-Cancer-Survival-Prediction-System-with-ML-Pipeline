import pandas as pd


def map_ui_inputs_to_features(age, tumor_size, tumor_stage, histologic_grade,
                               er_status, her2_status, pam50_subtype,
                               menopausal_state, oncotree_code, surgery_type):
    features = {
        "Age at Diagnosis":           age,
        "Tumor Size":                 tumor_size,
        "Tumor Stage":                tumor_stage,
        "Neoplasm Histologic Grade":  histologic_grade,
    }
    if er_status == "Positive":
        features["ER Status_Positive"] = 1
    if her2_status == "Positive":
        features["HER2 Status_Positive"] = 1

    features[f"Pam50 + Claudin-low subtype_{pam50_subtype}"] = 1

    if menopausal_state == "Pre":
        features["Inferred Menopausal State_Pre"] = 1

    features[f"Oncotree Code_{oncotree_code}"] = 1

    if surgery_type == "MASTECTOMY":
        features["Type of Breast Surgery_MASTECTOMY"] = 1

    return features


def format_shap_table(df):
    return (
        df[["feature", "value", "shap_value"]]
        .rename(columns={"feature": "Feature", "value": "Patient Value", "shap_value": "SHAP Impact"})
        .reset_index(drop=True)
    )
