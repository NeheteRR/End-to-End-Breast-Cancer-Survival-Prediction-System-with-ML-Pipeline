"""
tests/test_inference.py

Tests for the inference helpers and UI input mapping utilities.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.inference import build_patient_row
from app.utils.helpers      import map_ui_inputs_to_features


SAMPLE_FEATURE_COLS = [
    "Age at Diagnosis",
    "Tumor Size",
    "Tumor Stage",
    "Neoplasm Histologic Grade",
    "ER Status_Positive",
    "HER2 Status_Positive",
    "Pam50 + Claudin-low subtype_LumA",
    "Inferred Menopausal State_Pre",
    "Oncotree Code_IDC",
    "Type of Breast Surgery_MASTECTOMY",
]


def test_build_patient_row_shape():
    patient = {"Age at Diagnosis": 50, "Tumor Size": 22}
    row = build_patient_row(patient, SAMPLE_FEATURE_COLS)
    assert row.shape == (1, len(SAMPLE_FEATURE_COLS))


def test_build_patient_row_known_values():
    patient = {"Age at Diagnosis": 55, "ER Status_Positive": 1}
    row = build_patient_row(patient, SAMPLE_FEATURE_COLS)
    assert row.loc[0, "Age at Diagnosis"] == 55
    assert row.loc[0, "ER Status_Positive"] == 1


def test_build_patient_row_unknown_key_ignored(capsys):
    patient = {"UNKNOWN_FEATURE": 99}
    row = build_patient_row(patient, SAMPLE_FEATURE_COLS)
    captured = capsys.readouterr()
    assert "not found" in captured.out
    # All known columns should be zero-initialised
    assert (row.values == 0).all()


def test_map_ui_inputs_returns_dict():
    result = map_ui_inputs_to_features(
        age=50, tumor_size=22, tumor_stage=2, histologic_grade=3,
        er_status="Positive", her2_status="Negative",
        pam50_subtype="LumA", menopausal_state="Post",
        oncotree_code="IDC", surgery_type="MASTECTOMY",
    )
    assert isinstance(result, dict)
    assert result["Age at Diagnosis"] == 50
    assert result.get("ER Status_Positive") == 1
    assert "HER2 Status_Positive" not in result  # Negative should not appear


def test_map_ui_inputs_negative_er():
    result = map_ui_inputs_to_features(
        age=45, tumor_size=18, tumor_stage=1, histologic_grade=2,
        er_status="Negative", her2_status="Negative",
        pam50_subtype="Normal", menopausal_state="Pre",
        oncotree_code="ILC", surgery_type="BREAST CONSERVING",
    )
    assert "ER Status_Positive" not in result
    assert result.get("Inferred Menopausal State_Pre") == 1
    assert "Type of Breast Surgery_MASTECTOMY" not in result
