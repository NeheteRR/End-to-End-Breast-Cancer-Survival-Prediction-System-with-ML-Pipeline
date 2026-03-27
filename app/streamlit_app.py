import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shap
import streamlit as st
import streamlit.components.v1 as components

from app.services.model_io       import load_model, load_feature_columns, load_background
from app.services.inference      import predict
from app.services.explainability import get_explainer, explain_patient
from app.utils.helpers           import map_ui_inputs_to_features, format_shap_table
from config                      import PAM50_SUBTYPES, ONCOTREE_CODES, SURGERY_TYPES

st.set_page_config(page_title="Chemo Response Predictor", page_icon="🔬", layout="wide")
st.title("🔬 Chemotherapy Response Predictor")
st.markdown("Predicts whether a breast cancer patient is likely to **respond to chemotherapy** using a model trained on the METABRIC dataset.")


@st.cache_resource
def load_artefacts():
    model        = load_model()
    feature_cols = load_feature_columns()
    background   = load_background()
    explainer    = get_explainer(model, background)
    return model, feature_cols, explainer


try:
    model, feature_cols, explainer = load_artefacts()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()


with st.sidebar:
    st.header("Patient Clinical Inputs")
    age         = st.slider("Age at Diagnosis",    20, 90, 50)
    tumor_size  = st.slider("Tumor Size (mm)",     1, 200, 22)
    tumor_stage = st.selectbox("Tumor Stage",      [1, 2, 3, 4])
    grade       = st.selectbox("Histologic Grade", [1, 2, 3])
    er_status   = st.selectbox("ER Status",        ["Positive", "Negative"])
    her2_status = st.selectbox("HER2 Status",      ["Positive", "Negative"])
    pam50       = st.selectbox("PAM50 Subtype",    PAM50_SUBTYPES)
    menopause   = st.selectbox("Menopausal State", ["Pre", "Post"])
    oncotree    = st.selectbox("Oncotree Code",    ONCOTREE_CODES)
    surgery     = st.selectbox("Surgery Type",     SURGERY_TYPES)
    predict_btn = st.button("Run Prediction", use_container_width=True)


if predict_btn:
    feature_dict = map_ui_inputs_to_features(
        age=age, tumor_size=tumor_size, tumor_stage=tumor_stage,
        histologic_grade=grade, er_status=er_status, her2_status=her2_status,
        pam50_subtype=pam50, menopausal_state=menopause,
        oncotree_code=oncotree, surgery_type=surgery,
    )

    result      = predict(feature_dict, model=model, feature_cols=feature_cols)
    explanation = explain_patient(result["patient_row"], model, feature_cols, explainer)

    col1, col2 = st.columns(2)
    with col1:
        if result["prediction"] == 1:
            st.success(f"**Responder**  (p = {result['probability']:.3f})")
        else:
            st.error(f"**Non-Responder**  (p = {result['probability']:.3f})")
    with col2:
        st.metric("Response Probability", f"{result['probability']:.1%}")

    st.divider()
    st.subheader("SHAP Explainability")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Factors increasing response probability**")
        st.dataframe(format_shap_table(explanation["top_positive"]), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Factors decreasing response probability**")
        st.dataframe(format_shap_table(explanation["top_negative"]), use_container_width=True, hide_index=True)

    st.subheader("SHAP Force Plot")
    try:
        shap_vals_local = explainer.shap_values(result["patient_row"])
        if isinstance(shap_vals_local, list):
            vals     = shap_vals_local[1]
            expected = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            vals     = shap_vals_local
            expected = explainer.expected_value
        force_html = shap.force_plot(expected, vals, result["patient_row"], matplotlib=False)
        components.html(shap.getjs() + force_html.html(), height=200)
    except Exception:
        st.info("Force plot not available for this model type. SHAP tables above still show feature impact.")

else:
    st.info("Configure patient inputs in the sidebar and click **Run Prediction**.")
