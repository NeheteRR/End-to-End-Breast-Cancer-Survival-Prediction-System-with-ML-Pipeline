import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def impute(df):
    df = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        if col != TARGET_COLUMN:
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
    return df


def encode(df):
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y, X_encoded.columns.tolist()


def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=RANDOM_STATE)
    return sm.fit_resample(X_train, y_train)


def run_feature_pipeline(df):
    df = impute(df)
    X, y, feature_cols = encode(df)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, y_train = apply_smote(X_train, y_train)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_cols": feature_cols,
    }
