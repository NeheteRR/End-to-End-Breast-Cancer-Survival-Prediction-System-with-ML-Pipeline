import joblib
import pandas as pd
from config import MODEL_PATH, FEATURE_COLS_PATH

model        = joblib.load(MODEL_PATH)
feature_cols = pd.read_csv(FEATURE_COLS_PATH)["feature"].tolist()

# --- edit your patient here ---
patient = {
    "Age at Diagnosis":                      52,
    "Tumor Size":                            22,
    "Tumor Stage":                           2,
    "Neoplasm Histologic Grade":             3,
    "ER Status_Positive":                    1,   # 1 = Positive, remove if Negative
    "HER2 Status_Positive":                  0,
    "Pam50 + Claudin-low subtype_LumA":      1,
    "Inferred Menopausal State_Pre":         0,
    "Type of Breast Surgery_MASTECTOMY":     1,
    "Oncotree Code_IDC":                     1,
}

row = pd.DataFrame([{col: 0.0 for col in feature_cols}])
for key, val in patient.items():
    if key in row.columns:
        row[key] = val

prediction  = int(model.predict(row)[0])
probability = float(model.predict_proba(row)[0][1])
label       = "Responder" if prediction == 1 else "Non-Responder"

print(f"\nPrediction  : {label}")
print(f"Probability : {probability:.1%}")
