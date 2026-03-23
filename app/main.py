"""
app/main.py — Project entry point.

Usage:
    python app/main.py --mode train       # preprocess → train → evaluate → save
    python app/main.py --mode predict     # load model → predict example patient
    python app/main.py --mode evaluate    # reload saved model → re-evaluate on held-out test set
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.preprocessing     import run_preprocessing_pipeline
from app.services.feature_engineering import run_feature_pipeline
from app.services.model_trainer      import train_all
from app.services.evaluator          import evaluate_all, print_results
from app.services.model_io           import save_model, save_feature_columns, load_model, load_feature_columns
from app.services.inference          import predict
from app.services.explainability     import get_explainer, explain_patient
from app.utils.helpers               import map_ui_inputs_to_features
from config                          import RAW_DATA_PATH


# ── Example patient used by --mode predict ─────────────────────────────────
EXAMPLE_PATIENT_UI = dict(
    age=52,
    tumor_size=22,
    tumor_stage=2,
    histologic_grade=3,
    er_status="Positive",
    her2_status="Negative",
    pam50_subtype="LumA",
    menopausal_state="Post",
    oncotree_code="IDC",
    surgery_type="MASTECTOMY",
)


def run_training_pipeline() -> None:
    print("\n─── Step 1 / 4  Preprocessing ───────────────────────────────────")
    df = run_preprocessing_pipeline()
    print(f"  Dataset shape after preprocessing: {df.shape}")

    print("\n─── Step 2 / 4  Feature Engineering ─────────────────────────────")
    data = run_feature_pipeline(df)
    print(f"  Training samples (after SMOTE): {data['X_train'].shape[0]}")
    print(f"  Test samples                  : {data['X_test'].shape[0]}")
    print(f"  Feature count                 : {len(data['feature_cols'])}")

    print("\n─── Step 3 / 4  Training ─────────────────────────────────────────")
    models = train_all(data["X_train"], data["y_train"])

    print("\n─── Step 4 / 4  Evaluation & Saving ──────────────────────────────")
    results = evaluate_all(models, data["X_test"], data["y_test"])
    print_results(results)

    best_name  = results["_best_model_name"]
    best_model = models[best_name]
    save_model(best_model)
    save_feature_columns(data["feature_cols"])
    print(f"\n  Best model saved: {best_name}")


def run_prediction_example() -> None:
    feature_dict = map_ui_inputs_to_features(**EXAMPLE_PATIENT_UI)
    result = predict(feature_dict)

    label = "Responder" if result["prediction"] == 1 else "Non-Responder"
    print(f"\nPrediction  : {label}")
    print(f"Probability : {result['probability']:.3f}")

    # Local SHAP explanation
    model       = load_model()
    feat_cols   = load_feature_columns()
    explainer   = get_explainer(model)
    explanation = explain_patient(result["patient_row"], model, feat_cols, explainer)
    print(explanation["text_report"])


def main():
    parser = argparse.ArgumentParser(description="Chemo Response Predictor CLI")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        default="train",
        help="Pipeline mode (default: train)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training_pipeline()
    elif args.mode == "predict":
        run_prediction_example()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
