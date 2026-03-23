"""
app/models/schemas.py

Pydantic data models for request / response validation.
These are used by the Streamlit UI and can be reused by a FastAPI layer.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class PatientInput(BaseModel):
    """Clinical input features for a single breast cancer patient."""

    age_at_diagnosis:       float = Field(..., ge=0, le=120, description="Patient age at diagnosis (years)")
    tumor_size:             float = Field(..., ge=0,          description="Tumour size (mm)")
    tumor_stage:            int   = Field(..., ge=1, le=4,    description="Clinical tumour stage (1–4)")
    histologic_grade:       int   = Field(..., ge=1, le=3,    description="Nottingham histologic grade (1–3)")
    er_status:              Literal["Positive", "Negative"]
    her2_status:            Literal["Positive", "Negative"]
    pam50_subtype:          Literal["LumA", "LumB", "Her2", "Normal", "claudin-low"]
    menopausal_state:       Literal["Pre", "Post"]
    oncotree_code:          Literal["IDC", "ILC", "MBC", "MDLC", "IMMC"]
    surgery_type:           Literal["MASTECTOMY", "BREAST CONSERVING"]


class PredictionResult(BaseModel):
    """Model output for a single patient prediction."""

    prediction:       int   = Field(..., description="1 = Responder, 0 = Non-Responder")
    probability:      float = Field(..., description="Probability of chemotherapy response")
    label:            str   = Field(..., description="Human-readable prediction label")
    top_positive:     list  = Field(default_factory=list, description="Top features increasing probability")
    top_negative:     list  = Field(default_factory=list, description="Top features decreasing probability")
