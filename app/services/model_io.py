"""
app/services/model_io.py

Handles saving and loading of trained model artefacts:
  - Serialised sklearn/XGBoost model (.pkl via joblib)
  - Feature column order (.csv)
"""

import joblib
import pandas as pd
from pathlib import Path

from config import MODEL_PATH, FEATURE_COLS_PATH


def save_model(model, path: Path = MODEL_PATH) -> None:
    """Serialise a fitted model to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model(path: Path = MODEL_PATH):
    """Load a previously serialised model from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            "Run the training pipeline first: python app/main.py --mode train"
        )
    return joblib.load(path)


def save_feature_columns(feature_cols: list[str], path: Path = FEATURE_COLS_PATH) -> None:
    """Persist the ordered feature column list so inference can reconstruct the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_cols}).to_csv(path, index=False)
    print(f"Feature columns saved → {path}")


def load_feature_columns(path: Path = FEATURE_COLS_PATH) -> list[str]:
    """Reload the saved feature column list."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature columns file not found at {path}. "
            "Run the training pipeline first."
        )
    return pd.read_csv(path)["feature"].tolist()
