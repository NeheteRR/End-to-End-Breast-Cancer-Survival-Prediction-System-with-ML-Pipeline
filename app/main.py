import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.preprocessing    import run_preprocessing_pipeline
from app.services.feature_engineering import run_feature_pipeline
from app.services.model_trainer    import train_all
from app.services.evaluator        import evaluate_all
from app.services.model_io         import save_model, save_feature_columns
from config                        import ARTIFACTS_DIR

# Simpler entry: just run train.py or predict.py from the project root.
# This file is kept for backward compatibility.

def run_training_pipeline():
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
    save_model(best_model)
    save_feature_columns(data["feature_cols"])
    print(f"\nBest model saved: {best_name}")


if __name__ == "__main__":
    run_training_pipeline()
