"""
tests/test_preprocessing.py

Unit tests for the preprocessing and feature engineering services.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd

# Make project root importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.preprocessing import (
    clean_dataframe,
    create_chemo_response_label,
    drop_leakage_columns,
    select_features,
)
from app.services.feature_engineering import impute, encode


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_sample():
    """Minimal DataFrame that mimics the raw METABRIC clinical data."""
    return pd.DataFrame({
        "Age at Diagnosis":          [45.0, 60.0, 52.0, 37.0, 70.0],
        "Tumor Size":                ["22", "15,0", None,  "40", "10"],
        "Tumor Stage":               [2,    3,     1,    None,  2   ],
        "Neoplasm Histologic Grade": [3,    2,     1,    3,     2   ],
        "Overall Survival (Months)": [72,   30,    90,   45,    80  ],
        "Overall Survival Status":   ["1:DECEASED", "0:LIVING", "0", "1", "0"],
        "Relapse Free Status":       ["0:Not Recurred", "1", "0", "0", "0"],
        "Chemotherapy":              ["yes", "yes", "No", "yes", "Yes"],
        "ER Status":                 ["Positive", "Negative", "Positive", "Positive", "Negative"],
        "HER2 Status":               ["Negative", "Positive", "Negative", "Negative", "Positive"],
        "Pam50 + Claudin-low subtype": ["LumA", "Her2", "LumB", "LumA", "claudin-low"],
        "Inferred Menopausal State": ["Post", "Pre", "Post", "Pre", "Post"],
        "Oncotree Code":             ["IDC", "ILC", "IDC", "IDC", "MBC"],
        "Type of Breast Surgery":    ["MASTECTOMY", "BREAST CONSERVING", "MASTECTOMY",
                                      "MASTECTOMY", "BREAST CONSERVING"],
    })


# ── Tests: clean_dataframe ────────────────────────────────────────────────────

def test_clean_numeric_columns(raw_sample):
    df = clean_dataframe(raw_sample)
    assert df["Tumor Size"].dtype == float, "Tumor Size should be float after cleaning"


def test_clean_chemotherapy_binary(raw_sample):
    df = clean_dataframe(raw_sample)
    assert set(df["Chemotherapy"].dropna().unique()).issubset({0, 1})


def test_clean_survival_status(raw_sample):
    df = clean_dataframe(raw_sample)
    assert set(df["Overall Survival Status"].dropna().unique()).issubset({0, 1})


def test_clean_relapse_status(raw_sample):
    df = clean_dataframe(raw_sample)
    assert set(df["Relapse Free Status"].dropna().unique()).issubset({0, 1})


# ── Tests: create_chemo_response_label ────────────────────────────────────────

def test_label_creation_only_chemo_patients(raw_sample):
    df = clean_dataframe(raw_sample)
    labeled = create_chemo_response_label(df)
    # Only rows where Chemotherapy == 1 should remain
    assert labeled.shape[0] > 0
    # No rows without Chemotherapy == 1 should appear
    assert (labeled["Chemotherapy"] == 1).all()


def test_label_values_are_binary(raw_sample):
    df = clean_dataframe(raw_sample)
    labeled = create_chemo_response_label(df)
    assert set(labeled["Chemo_Response"].unique()).issubset({0.0, 1.0})


# ── Tests: drop_leakage_columns ───────────────────────────────────────────────

def test_leakage_columns_removed(raw_sample):
    df = clean_dataframe(raw_sample)
    labeled = create_chemo_response_label(df)
    cleaned = drop_leakage_columns(labeled)
    assert "Overall Survival (Months)" not in cleaned.columns
    assert "Relapse Free Status" not in cleaned.columns
    assert "Chemotherapy" not in cleaned.columns


# ── Tests: impute ─────────────────────────────────────────────────────────────

def test_impute_removes_nulls():
    df = pd.DataFrame({
        "A": [1.0, None, 3.0],
        "B": ["x", None, "z"],
        "Chemo_Response": [1, 0, 1],
    })
    result = impute(df)
    assert result[["A", "B"]].isna().sum().sum() == 0


# ── Tests: encode ─────────────────────────────────────────────────────────────

def test_encode_returns_numeric_only():
    df = pd.DataFrame({
        "Num": [1.0, 2.0, 3.0],
        "Cat": ["a", "b", "a"],
        "Chemo_Response": [1, 0, 1],
    })
    X, y, cols = encode(df)
    assert X.select_dtypes(include=["object"]).shape[1] == 0, "All columns should be numeric after encoding"


def test_encode_target_not_in_features():
    df = pd.DataFrame({
        "Num": [1.0, 2.0, 3.0],
        "Chemo_Response": [1, 0, 1],
    })
    X, y, cols = encode(df)
    assert "Chemo_Response" not in X.columns
