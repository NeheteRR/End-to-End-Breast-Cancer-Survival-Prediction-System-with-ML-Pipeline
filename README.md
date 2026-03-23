# 🔬 Chemotherapy Response Predictor

A production-ready ML system that predicts whether a breast cancer patient will
respond to chemotherapy, trained on the **METABRIC** clinical dataset.  
Includes a Streamlit web interface with SHAP-based explainability for every prediction.

---

## Features

- **End-to-end ML pipeline** — preprocessing → feature engineering → SMOTE → training → evaluation
- **3 classifiers** — Logistic Regression, Random Forest, XGBoost (best model auto-selected by AUC)
- **SHAP explainability** — global feature importance + per-patient waterfall / force plots
- **Streamlit UI** — interactive sidebar inputs, prediction banner, SHAP tables, force plot
- **Fully modular** — each concern lives in its own service module
- **Pydantic schemas** — validated input / output models ready for a REST API layer
- **14 unit tests** covering preprocessing, encoding, and inference

---

## Tech Stack

| Layer          | Library                                      |
|----------------|----------------------------------------------|
| Data           | pandas, numpy                                |
| ML             | scikit-learn, xgboost, imbalanced-learn      |
| Explainability | shap, matplotlib                             |
| UI             | streamlit                                    |
| Validation     | pydantic                                     |
| Testing        | pytest, pytest-cov                           |

---

## Project Structure

```
chemo-response-predictor/
├── app/
│   ├── main.py                  # CLI entry point (train / predict)
│   ├── streamlit_app.py         # Interactive Streamlit web UI
│   ├── services/
│   │   ├── preprocessing.py     # Load, clean, label creation
│   │   ├── feature_engineering.py  # Imputation, encoding, SMOTE
│   │   ├── model_trainer.py     # LR + RF + XGBoost training
│   │   ├── evaluator.py         # AUC, F1, confusion matrix
│   │   ├── model_io.py          # Save / load model artefacts
│   │   ├── inference.py         # Single-patient prediction
│   │   └── explainability.py    # SHAP global & local explanations
│   ├── models/
│   │   └── schemas.py           # Pydantic input / output schemas
│   └── utils/
│       └── helpers.py           # UI → feature mapping, formatters
├── artifacts/                   # Saved model & feature columns (gitignored)
├── data/                        # Raw METABRIC TSV (gitignored)
├── notebooks/                   # Original exploratory notebook
├── tests/
│   ├── test_preprocessing.py
│   └── test_inference.py
├── config.py                    # All constants & hyperparameters
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone https://github.com/your-username/chemo-response-predictor
cd chemo-response-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place the METABRIC TSV file at:

```
data/brca_metabric_clinical_data.tsv
```

Download it from [cBioPortal — METABRIC](https://www.cbioportal.org/study/summary?id=brca_metabric)
→ "Clinical Data" tab → download as TSV.

---

## How to Run

### 1. Train the model

```bash
python app/main.py --mode train
```

Runs the full pipeline (preprocessing → SMOTE → training → evaluation) and saves:
- `artifacts/best_chemo_response_model.pkl`
- `artifacts/feature_columns.csv`

### 2. Run a single prediction (CLI)

```bash
python app/main.py --mode predict
```

Loads the saved model, predicts for an example patient, and prints a SHAP text report.

### 3. Launch the Streamlit web app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the test suite

```bash
pytest tests/ -v --cov=app
```

---

## Model Details

| Step                  | Decision                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| Label engineering     | Patients surviving > 60 months post-chemo with no relapse = Responder   |
| Leakage prevention    | Survival & relapse columns dropped before any modelling                  |
| Class imbalance       | SMOTE applied on the training fold only (never on test data)             |
| Evaluation metric     | ROC-AUC primary; F1-score secondary                                      |
| Best model (default)  | Random Forest — 300 trees, max depth 10                                  |
| Explainability        | SHAP TreeExplainer — global bar/dot plots + per-patient force/waterfall  |

---

## Dataset

**METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium).  
Clinical data for ~2,500 breast cancer patients including tumour characteristics,
receptor status, treatment history, and survival outcomes.

Source: [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)

---

## Suggested Improvements

See the bottom of this README for a roadmap of enhancements that would make this
project even more impressive for a portfolio or production deployment:

1. **FastAPI REST layer** — expose `/predict` and `/explain` endpoints; Pydantic schemas are already in place.
2. **Docker + deployment** — add `Dockerfile` + `docker-compose.yml` and deploy to Railway/Render for a live demo URL.
3. **MLflow experiment tracking** — log hyperparameters, metrics, and model artefacts for every training run.
4. **Cross-validation + Optuna tuning** — replace the single split with `StratifiedKFold` + Bayesian hyperparameter search.
5. **GitHub Actions CI** — run `pytest` on every push; add a green badge to the README.

---
