import joblib
import pandas as pd
from app.services.preprocessing import run_preprocessing_pipeline
from app.services.feature_engineering import run_feature_pipeline
from app.services.model_trainer import train_all
from app.services.evaluator import evaluate_all
from config import MODEL_PATH, FEATURE_COLS_PATH, BACKGROUND_PATH, ARTIFACTS_DIR

ARTIFACTS_DIR.mkdir(exist_ok=True)

print("Preprocessing...")
df = run_preprocessing_pipeline()

print("Feature engineering + SMOTE...")
data = run_feature_pipeline(df)

print("Training models...")
models = train_all(data["X_train"], data["y_train"])

print("Evaluating...")
results = evaluate_all(models, data["X_test"], data["y_test"])

for name, m in results.items():
    if name.startswith("_"):
        continue
    print(f"  {name:20s}  AUC: {m['auc']}  F1: {m['f1']}")

best_name  = results["_best_model_name"]
best_model = models[best_name]
print(f"\nBest model: {best_name}")

# save 100 rows of training data as SHAP background (needed for LinearExplainer)
background = data["X_train"].sample(n=min(100, len(data["X_train"])), random_state=42)

joblib.dump(best_model, MODEL_PATH)
pd.DataFrame({"feature": data["feature_cols"]}).to_csv(FEATURE_COLS_PATH, index=False)
joblib.dump(background, BACKGROUND_PATH)

print(f"Saved → {MODEL_PATH}")
print(f"Saved → {FEATURE_COLS_PATH}")
print(f"Saved → {BACKGROUND_PATH}")
