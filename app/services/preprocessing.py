import re
import pandas as pd
import numpy as np
from config import (
    RAW_DATA_PATH, COLUMNS_TO_DROP, LEAKAGE_COLUMNS,
    NUMERIC_COLS_TO_CLEAN, FEATURE_COLUMNS, TARGET_COLUMN,
    SURVIVAL_THRESHOLD_MONTHS,
)


def _to_num(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(",", "")
        if s.lower() in {"none", "na", "nan", "", "-", "--", "null"}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def _extract_status_digit(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    match = re.search(r"\b([01])\b", s)
    if match:
        return int(match.group(1))
    try:
        return int(float(s))
    except Exception:
        return np.nan


def _clean_yes_no(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    mapping = {
        "yes": 1, "y": 1, "1": 1, "true": 1, "t": 1,
        "no":  0, "n": 0, "0": 0, "false": 0, "f": 0,
    }
    if s in mapping:
        return mapping[s]
    if "yes" in s:
        return 1
    if "no" in s:
        return 0
    return np.nan


def load_raw_data(path=None):
    path = path or RAW_DATA_PATH
    df = pd.read_csv(path, sep="\t")
    drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=drop)


def clean_dataframe(df):
    df = df.copy()
    for col, fn in [
        ("Overall Survival Status", _extract_status_digit),
        ("Relapse Free Status",     _extract_status_digit),
        ("Chemotherapy",            _clean_yes_no),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(fn)
    for col in NUMERIC_COLS_TO_CLEAN:
        if col in df.columns:
            df[col] = df[col].apply(_to_num)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    return df


def create_chemo_response_label(df):
    df = df.copy()
    cond_chemo      = df.get("Chemotherapy") == 1
    cond_os         = df.get("Overall Survival (Months)") > SURVIVAL_THRESHOLD_MONTHS
    cond_no_relapse = df.get("Relapse Free Status") == 0

    df[TARGET_COLUMN] = np.nan
    df.loc[cond_chemo &  (cond_os & cond_no_relapse), TARGET_COLUMN] = 1
    df.loc[cond_chemo & ~(cond_os & cond_no_relapse), TARGET_COLUMN] = 0

    return df[df[TARGET_COLUMN].notna()].copy()


def drop_leakage_columns(df):
    return df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])


def select_features(df):
    keep = [c for c in FEATURE_COLUMNS if c in df.columns] + [TARGET_COLUMN]
    return df[keep].copy()


def run_preprocessing_pipeline(path=None):
    df = load_raw_data(path)
    df = clean_dataframe(df)
    df = create_chemo_response_label(df)
    df = drop_leakage_columns(df)
    return select_features(df)
