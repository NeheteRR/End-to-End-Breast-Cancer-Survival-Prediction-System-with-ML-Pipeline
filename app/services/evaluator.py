"""
app/services/evaluator.py

Computes evaluation metrics for every trained model and returns a
structured results dict — no side effects (no file writes, no prints).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str = "") -> dict:
    """
    Evaluate a single fitted classifier.

    Returns a dict with:
      - name, auc, f1, confusion_matrix, classification_report (text)
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = None

    return {
        "name":                  name,
        "auc":                   round(auc, 4) if auc is not None else None,
        "f1":                    round(f1_score(y_test, preds), 4),
        "confusion_matrix":      confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, digits=4),
    }


def evaluate_all(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate every model in *models* dict and return a summary dict.

    Also determines the best model by AUC (falls back to F1).
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

    # Pick best model
    best_name = max(
        results,
        key=lambda n: results[n]["auc"] if results[n]["auc"] is not None else results[n]["f1"],
    )
    results["_best_model_name"] = best_name
    return results


def print_results(results: dict) -> None:
    """Pretty-print evaluation results to stdout."""
    for name, metrics in results.items():
        if name.startswith("_"):
            continue
        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"{'='*40}")
        print(f"  AUC : {metrics['auc']}")
        print(f"  F1  : {metrics['f1']}")
        print(f"  Confusion Matrix : {metrics['confusion_matrix']}")
        print(metrics["classification_report"])
    print(f"\n★  Best model: {results.get('_best_model_name')}")
