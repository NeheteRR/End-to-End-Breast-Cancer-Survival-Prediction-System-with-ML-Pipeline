import pandas as pd
from app.services.model_io import load_model, load_feature_columns


def build_patient_row(patient_dict, feature_cols):
    row = pd.DataFrame([{col: 0.0 for col in feature_cols}])
    for key, val in patient_dict.items():
        if key in feature_cols:
            row.loc[0, key] = val
        else:
            print(f"  Warning: '{key}' not found in model schema — skipped.")
    return row


def predict(patient_dict, model=None, feature_cols=None):
    if model is None:
        model = load_model()
    if feature_cols is None:
        feature_cols = load_feature_columns()

    row         = build_patient_row(patient_dict, feature_cols)
    prediction  = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    return {"prediction": prediction, "probability": round(probability, 4), "patient_row": row}
