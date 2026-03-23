"""
config.py — Centralised project configuration.
All hard-coded paths, model hyper-parameters and thresholds live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

RAW_DATA_PATH = DATA_DIR / "brca_metabric_clinical_data.tsv"
MODEL_PATH = ARTIFACTS_DIR / "best_chemo_response_model.pkl"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_columns.csv"

# ── Data cleaning ──────────────────────────────────────────────────────────
COLUMNS_TO_DROP = [
    "Patient's Vital Status",
    "Hormone Therapy",
    "Radio Therapy",
    "Cohort",
    "Number of Samples Per Patient",
]

LEAKAGE_COLUMNS = [
    "Overall Survival (Months)",
    "Overall Survival Status",
    "Relapse Free Status (Months)",
    "Relapse Free Status",
    "Chemotherapy",
    # also listed in COLUMNS_TO_DROP but safe to repeat
    "Patient's Vital Status",
    "Hormone Therapy",
    "Radio Therapy",
    "Cohort",
    "Number of Samples Per Patient",
]

NUMERIC_COLS_TO_CLEAN = [
    "Overall Survival (Months)",
    "Relapse Free Status (Months)",
    "Tumor Size",
    "Tumor Stage",
    "Age at Diagnosis",
    "Mutation Count",
    "TMB (nonsynonymous)",
    "Lymph nodes examined positive",
    "Nottingham prognostic index",
]

FEATURE_COLUMNS = [
    "Age at Diagnosis",
    "Tumor Size",
    "Tumor Stage",
    "Neoplasm Histologic Grade",
    "Lymph nodes examined positive",
    "Nottingham prognostic index",
    "Mutation Count",
    "TMB (nonsynonymous)",
    "Type of Breast Surgery",
    "Cancer Type Detailed",
    "Cellularity",
    "Pam50 + Claudin-low subtype",
    "ER status measured by IHC",
    "ER Status",
    "PR Status",
    "HER2 status measured by SNP6",
    "HER2 Status",
    "Tumor Other Histologic Subtype",
    "Inferred Menopausal State",
    "Integrative Cluster",
    "Primary Tumor Laterality",
    "Oncotree Code",
    "Sample Type",
    "Sex",
    "3-Gene classifier subtype",
]

TARGET_COLUMN = "Chemo_Response"

# ── Label-creation thresholds ──────────────────────────────────────────────
SURVIVAL_THRESHOLD_MONTHS = 60   # patients surviving > 60 months → responder

# ── Train / test split ─────────────────────────────────────────────────────
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ── Model hyper-parameters ─────────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
}

# ── Streamlit UI options ───────────────────────────────────────────────────
PAM50_SUBTYPES = ["LumA", "LumB", "Her2", "Normal", "claudin-low"]
ONCOTREE_CODES = ["IDC", "ILC", "MBC", "MDLC", "IMMC"]
SURGERY_TYPES = ["MASTECTOMY", "BREAST CONSERVING"]
