# 🔬 Chemotherapy Response Predictor

A complete end-to-end Machine Learning web application designed to predict whether a breast cancer patient will respond positively to chemotherapy. Trained on the METABRIC clinical dataset, this project features robust model selection, a production-ready prediction pipeline, and a Streamlit-powered dashboard complete with SHAP-based model explainability.

---

## 🚀 Key Features

- **Automated ML Pipeline**: Preprocessing, Feature Engineering, SMOTE balancing, and training of `LogisticRegression`, `RandomForest`, and `XGBoost`.
- **Model Explainability**: Native integration with `SHAP` values. Features both global and individual decision insights on predictions.
- **RESTful-like Prediction CLI**: Single-patient CLI scripts tailored for rapid integration.
- **Interactive UI**: A sleek, user-friendly Streamlit dashboard to interactively test patient parameters.

---

## 🛠 Tech Stack

| Component               | Technologies                               |
|-------------------------|--------------------------------------------|
| **Data Manipulation**   | Pandas, NumPy                              |
| **Machine Learning**    | Scikit-Learn, XGBoost, Imbalanced-Learn    |
| **Model Explainability**| SHAP, Matplotlib                           |
| **Web Interface**       | Streamlit                                  |
| **Validation**          | Pydantic                                   |
| **Testing**             | Pytest                                     |

---

## 📂 Project Structure

```text
chemo-response-predictor/
├── train.py                        # Executable: Trains models and generates artifacts
├── predict.py                      # Executable: Generates prediction for a local patient
├── config.py                       # Configuration: Paths, columns, and hyperparameters
├── requirements.txt                # Python dependencies
│
├── app/                            # Application Layer
│   ├── streamlit_app.py            # Streamlit logic and dashboard UI
│   ├── services/
│   │   ├── preprocessing.py        # Loading, cleaning, and label building
│   │   ├── feature_engineering.py  # Imputation, SME, transformations, SMOTE
│   │   ├── model_trainer.py        # Algorithm configurations and training routines
│   │   ├── evaluator.py            # AUC/F1 evaluations and best model selection
│   │   ├── model_io.py             # Saving & loading `.pkl` artifacts
│   │   ├── inference.py            # Local inference predictions
│   │   └── explainability.py       # SHAP Explainer orchestrations
│   ├── models/
│   │   └── schemas.py              # Pydantic schemas for IO validation
│   └── utils/
│       └── helpers.py              # Helper mapping utilities and UI formatters
│
├── artifacts/                      # Auto-generated model artifacts (.pkl, .csv)
├── data/                           # (Required) brca_metabric_clinical_data.tsv
└── tests/                          # Pytest suite
```

---

## ⚙️ Setup & Installation

**Prerequisite:** Python 3.9+ installed on your machine.

1. **Clone the repository and enter the directory**
   ```bash
   git clone <your-repo-url>
   cd chemo-response-predictor
   ```

2. **Create a virtual environment and activate it**
   ```bash
   python -m venv venv

   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Dataset**
   Place your `brca_metabric_clinical_data.tsv` file inside the `data/` directory.

---

## 💡 Usage Guide

### 1. Train the Model
You must run the training script first to generate the necessary artifacts (`model.pkl`, `feature_cols.csv`, `background.pkl`).
```bash
python train.py
```
*This will evaluate Logistic Regression, Random Forest, and XGBoost, pick the best one by AUC, and output it to the `artifacts/` folder.*

### 2. Predict on a Single Patient (CLI)
You can test the trained model locally on sample patient configurations via the CLI.
```bash
python predict.py
```
*(You can modify the `patient_data` dictionary directly inside `predict.py`.)*

### 3. Launch the Web Interface (Streamlit)
Start the visual dashboard to test various inputs using interactive sliders and dropdowns.
```bash
streamlit run app/streamlit_app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🧪 Running Tests
To ensure system integrity, run the Pytest suite:
```bash
pytest tests/ -v
```

---

## 📊 How the Target Label is Built

A patient is classified as a **Responder (1)** if the dataset meets **all** the following criteria:
- Received chemotherapy.
- Survived more than 60 months.
- Experienced no relapse.

Patients who received chemotherapy but do not meet the above survival constraints are labeled as **Non-Responders (0)**. *Note: Patients who did not receive chemotherapy at all are excluded prior to model training.*
