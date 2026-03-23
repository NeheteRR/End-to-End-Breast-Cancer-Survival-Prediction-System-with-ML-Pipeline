"""
app/services/feature_engineering.py

Handles imputation, encoding, and train/test splitting.
Keeps sklearn transformations explicit so they can be reproduced
at inference time without fitting again.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple median imputation for numeric columns and mode imputation for
    categorical columns.  Target column is left untouched.
    """
    df = df.copy()

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != TARGET_COLUMN
    ]
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        mode_vals = df[col].mode(dropna=True)
        fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)

    return df


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    One-hot encode categorical features and separate X / y.

    Returns:
        X_encoded   — feature DataFrame (all numeric after encoding)
        y           — integer Series for the target
        feature_cols — ordered list of column names (needed at inference time)
    """
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_cols = X_encoded.columns.tolist()
    return X_encoded, y, feature_cols


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE):
    """
    Oversample the minority class on the training set using SMOTE.
    SMOTE must only ever be applied AFTER the train/test split to prevent leakage.
    """
    sm = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def run_feature_pipeline(df: pd.DataFrame):
    """
    Full feature-engineering pipeline: impute → encode → split → SMOTE.

    Returns a dict with all split arrays plus the feature column list.
    """
    df = impute(df)
    X, y, feature_cols = encode(df)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return {
        "X_train": X_train_res,
        "X_test": X_test,
        "y_train": y_train_res,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }
