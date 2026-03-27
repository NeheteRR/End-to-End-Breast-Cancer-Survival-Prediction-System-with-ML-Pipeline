import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test, name=""):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    try:
        auc = round(roc_auc_score(y_test, probs), 4)
    except ValueError:
        auc = None
    return {
        "name":                  name,
        "auc":                   auc,
        "f1":                    round(f1_score(y_test, preds), 4),
        "confusion_matrix":      confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, digits=4),
    }


def evaluate_all(models, X_test, y_test):
    results = {name: evaluate_model(model, X_test, y_test, name) for name, model in models.items()}
    best = max(results, key=lambda n: results[n]["auc"] if results[n]["auc"] else results[n]["f1"])
    results["_best_model_name"] = best
    return results
