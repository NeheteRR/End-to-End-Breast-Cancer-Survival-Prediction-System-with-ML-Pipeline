"""
app/services/model_trainer.py

Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Returns trained model objects; persistence is handled separately in model_io.py.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)


def build_models() -> dict:
    """Instantiate all three classifiers with config-driven hyper-parameters."""
    return {
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "RandomForest":       RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        "XGBoost":            XGBClassifier(**XGBOOST_PARAMS),
    }


def train_all(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fit every model in the catalogue on the (SMOTE-resampled) training data.

    Returns:
        dict mapping model name → fitted estimator
    """
    models = build_models()
    fitted = {}
    for name, model in models.items():
        print(f"  Training {name} …", flush=True)
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  ✓ {name} done.")
    return fitted
