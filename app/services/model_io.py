import joblib
import pandas as pd
from config import MODEL_PATH, FEATURE_COLS_PATH, BACKGROUND_PATH


def save_model(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Run train.py first.")
    return joblib.load(path)


def save_feature_columns(feature_cols, path=FEATURE_COLS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_cols}).to_csv(path, index=False)


def load_feature_columns(path=FEATURE_COLS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No feature columns found at {path}. Run train.py first.")
    return pd.read_csv(path)["feature"].tolist()


def load_background(path=BACKGROUND_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No background data found at {path}. Run train.py first.")
    return joblib.load(path)
