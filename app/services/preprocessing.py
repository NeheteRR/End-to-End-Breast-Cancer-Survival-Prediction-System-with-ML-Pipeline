"""
app/services/preprocessing.py

Handles all raw-data ingestion and cleaning steps:
  1. Load the METABRIC TSV
  2. Drop irrelevant / leaky columns
  3. Parse and normalise numeric / binary / categorical columns
  4. Engineer the Chemo_Response target label
  5. Filter to chemo-treated patients only
"""

import re
import pandas as pd
import numpy as np

from config import (
    RAW_DATA_PATH,
    COLUMNS_TO_DROP,
    LEAKAGE_COLUMNS,
    NUMERIC_COLS_TO_CLEAN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SURVIVAL_THRESHOLD_MONTHS,
)


# ── Low-level type-coercion helpers ───────────────────────────────────────

def _to_num(x) -> float:
    """Coerce a value to float, returning NaN on any failure."""
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
    """Extract a 0/1 status flag that may be encoded as '0:LIVING', '1', etc."""
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
    """Map free-text yes/no variants to 1/0."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    mapping = {
        "yes": 1, "y": 1, "1": 1, "true": 1, "t": 1,
        "no": 0,  "n": 0, "0": 0, "false": 0, "f": 0,
    }
    if s in mapping:
        return mapping[s]
    if "yes" in s:
        return 1
    if "no" in s:
        return 0
    return np.nan


# ── Public API ────────────────────────────────────────────────────────────

def load_raw_data(path=None) -> pd.DataFrame:
    """Load the METABRIC TSV file and return a raw DataFrame."""
    path = path or RAW_DATA_PATH
    df = pd.read_csv(path, sep="\t")
    # Drop known irrelevant columns immediately
    drop_existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=drop_existing)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all type-coercion and normalisation rules.
    Returns a cleaned copy; does NOT modify the input in place.
    """
    df = df.copy()

    # Binary status columns
    for col, fn in [
        ("Overall Survival Status", _extract_status_digit),
        ("Relapse Free Status",     _extract_status_digit),
        ("Chemotherapy",            _clean_yes_no),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(fn)

    # Numeric columns
    for col in NUMERIC_COLS_TO_CLEAN:
        if col in df.columns:
            df[col] = df[col].apply(_to_num)

    # Strip whitespace from all string/categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    return df


def create_chemo_response_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the binary Chemo_Response target:
      1 = treated with chemo AND survived > threshold AND no relapse
      0 = treated with chemo AND did NOT meet the above criteria

    Rows without chemotherapy treatment are discarded.
    """
    df = df.copy()

    cond_chemo     = df.get("Chemotherapy") == 1
    cond_os        = df.get("Overall Survival (Months)") > SURVIVAL_THRESHOLD_MONTHS
    cond_no_relapse = df.get("Relapse Free Status") == 0

    df[TARGET_COLUMN] = np.nan
    df.loc[cond_chemo &  (cond_os & cond_no_relapse), TARGET_COLUMN] = 1
    df.loc[cond_chemo & ~(cond_os & cond_no_relapse), TARGET_COLUMN] = 0

    # Keep only chemo-treated patients with a valid label
    df_chemo = df[df[TARGET_COLUMN].notna()].copy()
    return df_chemo


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that were used to construct the label (prevents data leakage)."""
    to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=to_drop)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the pre-defined modelling feature set plus the target column."""
    keep = [c for c in FEATURE_COLUMNS if c in df.columns]
    keep.append(TARGET_COLUMN)
    return df[keep].copy()


def run_preprocessing_pipeline(path=None) -> pd.DataFrame:
    """
    End-to-end preprocessing convenience function.

    Returns a clean, label-annotated DataFrame ready for feature engineering.
    """
    df = load_raw_data(path)
    df = clean_dataframe(df)
    df = create_chemo_response_label(df)
    df = drop_leakage_columns(df)
    df = select_features(df)
    return df
