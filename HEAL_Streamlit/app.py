"""
HEAL — Clinical Decision-Support Research Prototype
Hybrid Explainable Adaptive Learning for Stroke Risk Assessment
Explainable Machine-Learning Clinical Decision-Support Prototype
Locked Benchmark: Phase 8B Logistic Regression Pipeline (Threshold = 0.50)
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & HIGH-CONTRAST TYPOGRAPHY STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HEAL — Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS: Ensures crisp, dark, easily readable text in all browsers and lighting conditions
st.markdown(r"""
<style>
    /* Enforce crisp, high-contrast typography across the entire app */
    html, body, [class*="css"], .stApp {
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        background-color: #ffffff;
    }
    
    /* Headers: Bold, clear, strong contrast */
    h1 {
        color: #0f172a !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem !important;
    }
    h2 {
        color: #0f172a !important;
        font-size: 1.65rem !important;
        font-weight: 750 !important;
        margin-top: 0.6rem !important;
    }
    h3 {
        color: #0f172a !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }
    h4 {
        color: #1e293b !important;
        font-size: 1.15rem !important;
        font-weight: 650 !important;
    }
    
    /* Paragraphs and general text */
    p, span, label, div {
        color: #1e293b;
    }
    
    /* Form & Input Labels: Crisp, prominent, high readability */
    .stNumberInput label, .stSelectbox label, .stRadio label, .stTextInput label {
        color: #0f172a !important;
        font-size: 1.0rem !important;
        font-weight: 650 !important;
        margin-bottom: 6px !important;
    }
    
    /* Input fields: Strong text and borders */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        color: #0f172a !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        background-color: #f8fafc !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 6px !important;
    }
    
    /* Disclaimer Banner: High contrast */
    .disclaimer-box {
        background-color: #fefce8;
        border: 1.5px solid #fde047;
        border-left: 5px solid #ca8a04;
        color: #713f12 !important;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 22px;
        font-size: 0.95rem !important;
        line-height: 1.5;
        font-weight: 500;
    }
    .disclaimer-box strong {
        color: #713f12 !important;
    }
    
    /* Assessment Result Status Badges */
    .badge-elevated {
        background-color: #fee2e2;
        color: #991b1b !important;
        border: 1.5px solid #ef4444;
        padding: 8px 16px;
        border-radius: 18px;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        display: inline-block;
    }
    
    .badge-standard {
        background-color: #e0f2fe;
        color: #0369a1 !important;
        border: 1.5px solid #0284c7;
        padding: 8px 16px;
        border-radius: 18px;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        display: inline-block;
    }
    
    /* Score Output Tiles */
    .score-tile {
        background-color: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 8px;
        padding: 22px 24px;
        text-align: left;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }
    .score-value {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        color: #0284c7 !important;
        line-height: 1.1;
        margin: 6px 0;
    }
    .score-label {
        font-size: 0.88rem !important;
        color: #475569 !important;
        text-transform: uppercase;
        font-weight: 700 !important;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    
    /* Contributing Factor Items */
    .factor-item-pos {
        color: #7f1d1d !important;
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
    }
    .factor-item-neg {
        color: #1e3a8a !important;
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
    }

    /* Buttons: Clear, high contrast */
    .stButton > button {
        font-size: 1.02rem !important;
        font-weight: 650 !important;
        border-radius: 6px !important;
        padding: 8px 18px !important;
        color: #0f172a;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #0f172a !important;
        font-size: 0.96rem !important;
        font-weight: 550 !important;
    }

    /* Captions */
    .stCaption, small {
        color: #475569 !important;
        font-size: 0.90rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CACHED MODEL & ARTIFACT LOADERS (Single Load, Zero Retraining)
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_heal_model():
    """Load locked Phase 8B pipeline, specifications, and background data once."""
    model_path = os.path.join(APP_DIR, "model", "final_model_pipeline_phase8b.joblib")
    features_path = os.path.join(APP_DIR, "model", "final_features_phase8b.json")
    threshold_path = os.path.join(APP_DIR, "model", "final_threshold_phase8b.json")
    bg_path = os.path.join(APP_DIR, "model", "shap_background.json")
    params_path = os.path.join(APP_DIR, "model", "pipeline_parameters.json")
    
    pipeline = None
    if os.path.exists(model_path):
        try:
            pipeline = joblib.load(model_path)
        except Exception:
            pipeline = None
            
    with open(features_path) as f:
        feature_spec = json.load(f)
    with open(threshold_path) as f:
        threshold_spec = json.load(f)
    with open(bg_path) as f:
        shap_bg = json.load(f)
        
    pipeline_params = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            pipeline_params = json.load(f)
    else:
        # Fallback exact specifications for locked Phase 8B model
        pipeline_params = {
            'numerical_features': ['Systolic_BP', 'Diastolic_BP', 'BMI', 'HbA1c', 'Age'],
            'numerical_imputer_medians': [120.33333333333331, 74.0, 28.5, 5.5, 57.0],
            'scaler_means': [122.9609691023324, 74.90713486530683, 29.930338239091142, 5.769171185127808, 53.64626904208624],
            'scaler_scales': [17.94444449650798, 10.827176076409247, 7.454371151610486, 1.0930476536376221, 17.20902756223949],
            'categorical_features': ['Diabetes', 'Smoker', 'Sex', 'Hypertension', 'Heart_Disease'],
            'categorical_imputer_modes': [0.0, 0.0, 1.0, 0.0, 0.0],
            'coefficients': [-0.14924474069544763, 0.3605855494814883, -0.07751635267984662, 0.06705288274023681, 0.9219336023074952, 0.5517062709704036, 0.7976980882121566, -0.12109656272468144, 0.7559855176633208, 1.118613542186424],
            'intercept': -1.528966988287614
        }
        
    return pipeline, feature_spec, threshold_spec, shap_bg, pipeline_params

@st.cache_data
def load_deep_dive_assets():
    """Load empirical validation tables and performance curve data."""
    assets_dir = os.path.join(APP_DIR, "assets")
    
    metrics_8b_path = os.path.join(assets_dir, "final_metrics_phase8b.csv")
    metrics_p9_path = os.path.join(assets_dir, "phase9_generalization_metrics.csv")
    shap_global_path = os.path.join(assets_dir, "shap_global_feature_importance_phase10.csv")
    curves_path = os.path.join(assets_dir, "evaluation_curves.json")
    
    df_8b = pd.read_csv(metrics_8b_path) if os.path.exists(metrics_8b_path) else None
    df_p9 = pd.read_csv(metrics_p9_path) if os.path.exists(metrics_p9_path) else None
    df_shap = pd.read_csv(shap_global_path) if os.path.exists(shap_global_path) else None
    
    curves = {}
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            curves = json.load(f)
            
    return df_8b, df_p9, df_shap, curves

pipeline_8b, feature_spec, threshold_spec, shap_bg, pipeline_params = load_heal_model()
df_metrics_8b, df_metrics_p9, df_shap_global, evaluation_curves = load_deep_dive_assets()

LOCKED_THRESHOLD = float(threshold_spec.get('locked_threshold', 0.50))
MODEL_FEATURES = feature_spec['features']
NUMERICAL_FEATURES = feature_spec['numerical_features']
CATEGORICAL_FEATURES = feature_spec['categorical_features']

# -----------------------------------------------------------------------------
# 3. CORE INFERENCE & LOCAL SHAP FUNCTIONS
# -----------------------------------------------------------------------------
def predict_patient(patient_dict):
    """
    Execute inference strictly using Phase 8B feature order.
    Robust against cross-version environment incompatibilities (Python 3.9-3.14+, scikit-learn).
    Returns: prob (float), pred_class (int), transformed_df (pd.DataFrame)
    """
    input_df = pd.DataFrame([patient_dict])[MODEL_FEATURES]
    feat_order = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    # 1. Standard pipeline execution
    if pipeline_8b is not None:
        try:
            prob = float(pipeline_8b.predict_proba(input_df)[0, 1])
            pred_class = 1 if prob >= LOCKED_THRESHOLD else 0
            preprocessor = pipeline_8b.named_steps['preprocessor']
            transformed_vals = preprocessor.transform(input_df)
            transformed_df = pd.DataFrame(transformed_vals, columns=feat_order)
            return prob, pred_class, transformed_df
        except (AttributeError, Exception):
            pass

    # 2. Resilient cross-version exact inference (guaranteed 100% parity with Phase 8B pipeline)
    num_imputers = pipeline_params['numerical_imputer_medians']
    num_means = np.array(pipeline_params['scaler_means'])
    num_scales = np.array(pipeline_params['scaler_scales'])
    cat_imputers = pipeline_params['categorical_imputer_modes']
    coefs = np.array(pipeline_params['coefficients'])
    intercept = float(pipeline_params['intercept'])
    
    num_raw = np.array([
        patient_dict[c] if pd.notna(patient_dict.get(c)) else num_imputers[i]
        for i, c in enumerate(NUMERICAL_FEATURES)
    ], dtype=float)
    num_scaled = (num_raw - num_means) / num_scales
    
    cat_raw = np.array([
        patient_dict[c] if pd.notna(patient_dict.get(c)) else cat_imputers[i]
        for i, c in enumerate(CATEGORICAL_FEATURES)
    ], dtype=float)
    
    trans_vec = np.concatenate([num_scaled, cat_raw])
    transformed_df = pd.DataFrame([trans_vec], columns=feat_order)
    
    logit = intercept + float(np.dot(coefs, trans_vec))
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    pred_class = 1 if prob >= LOCKED_THRESHOLD else 0
    
    return prob, pred_class, transformed_df


def compute_local_shap(transformed_df):
    """
    Calculates exact local additive SHAP attributions:
    phi_i = beta_i * (x_i - mu_i)
    Matches Phase 10 LinearExplainer with development background masking.
    """
    feat_order = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    x_vec = transformed_df.iloc[0]
    
    shap_records = []
    base_val = shap_bg['base_value']
    
    for feat in feat_order:
        beta = shap_bg['coefficients'][feat]
        mu = shap_bg['mean_transformed'][feat]
        x_val = x_vec[feat]
        shap_val = beta * (x_val - mu)
        shap_records.append({
            'Feature': feat,
            'SHAP Value': shap_val,
            'Transformed Value': x_val,
            'Background Mean': mu,
            'Beta': beta
        })
        
    shap_df = pd.DataFrame(shap_records)
    shap_df['Abs Impact'] = shap_df['SHAP Value'].abs()
    shap_df = shap_df.sort_values(by='Abs Impact', ascending=False).reset_index(drop=True)
    return base_val, shap_df

# Friendly display names for clean lay-friendly UI
FEATURE_LABELS = {
    'Age': 'Age',
    'Systolic_BP': 'Systolic Blood Pressure',
    'Diastolic_BP': 'Diastolic Blood Pressure',
    'BMI': 'Body Mass Index (BMI)',
    'HbA1c': 'Blood Sugar (HbA1c)',
    'Diabetes': 'Diabetes Status',
    'Smoker': 'Smoking History',
    'Sex': 'Biological Sex',
    'Hypertension': 'Hypertension (High Blood Pressure)',
    'Heart_Disease': 'Heart Disease History'
}

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
view_mode = st.sidebar.radio(
    "Application Mode",
    ["Clinical Assessment", "Model Details & Scientific Validation"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Governance Status:**  
• Model: **Locked Phase 8B Logistic Regression**  
• Decision Threshold: **0.50**  
• Dataset: **CDC NHANES 2021–2023**  
• Explainability: **Exact SHAP Attribution**
""")
st.sidebar.caption("Clinical decision-support prototype. Not for commercial diagnosis.")

# App Header
st.markdown("# HEAL")
st.markdown("### **Hybrid Explainable Adaptive Learning for Stroke Risk Assessment**")
st.markdown("<p style='font-size: 1.05rem; color: #334155; font-weight: 500; margin-top: -6px;'>Explainable machine-learning clinical decision-support prototype</p>", unsafe_allow_html=True)

# Regulatory Disclaimer Banner
st.markdown(r"""
<div class="disclaimer-box">
    <strong>Research Prototype Notice:</strong> HEAL is an explainable decision-support research prototype trained on cross-sectional CDC NHANES data. 
    It evaluates statistical alignment with <em>reported history of stroke</em>. It does <strong>not</strong> provide a medical diagnosis, 
    does <strong>not</strong> forecast future incident stroke, and does <strong>not</strong> replace professional clinical judgment.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 1: CLINICAL ASSESSMENT (STREAMLINED SINGLE-SCROLL WORKFLOW)
# -----------------------------------------------------------------------------
if view_mode == "Clinical Assessment":
    st.markdown("---")
    
    # Initialize session state inputs if not set
    if 'patient_inputs' not in st.session_state:
        st.session_state['patient_inputs'] = {
            'Age': 58.0, 'Sex': 1.0, 'Systolic_BP': 124.0, 'Diastolic_BP': 78.0,
            'BMI': 28.5, 'HbA1c': 5.6, 'Diabetes': 0.0, 'Hypertension': 0.0,
            'Heart_Disease': 0.0, 'Smoker': 0.0
        }

    # Preset profiles toolbar
    st.markdown("#### Sample Clinical Profiles")
    pr_c1, pr_c2, pr_c3 = st.columns(3)
    
    with pr_c1:
        if st.button("Load Lower Model Score Case", use_container_width=True):
            st.session_state['patient_inputs'] = {
                'Age': 36.0, 'Sex': 1.0, 'Systolic_BP': 114.0, 'Diastolic_BP': 72.0,
                'BMI': 23.2, 'HbA1c': 5.2, 'Diabetes': 0.0, 'Hypertension': 0.0,
                'Heart_Disease': 0.0, 'Smoker': 0.0
            }
            st.rerun()
            
    with pr_c2:
        if st.button("Load Higher Model Score Case", use_container_width=True):
            st.session_state['patient_inputs'] = {
                'Age': 74.0, 'Sex': 0.0, 'Systolic_BP': 148.0, 'Diastolic_BP': 84.0,
                'BMI': 31.0, 'HbA1c': 6.8, 'Diabetes': 1.0, 'Hypertension': 1.0,
                'Heart_Disease': 1.0, 'Smoker': 1.0
            }
            st.rerun()
            
    with pr_c3:
        if st.button("Load Threshold Boundary Case (0.50)", use_container_width=True):
            st.session_state['patient_inputs'] = {
                'Age': 66.0, 'Sex': 1.0, 'Systolic_BP': 130.0, 'Diastolic_BP': 76.0,
                'BMI': 29.0, 'HbA1c': 5.8, 'Diabetes': 0.0, 'Hypertension': 1.0,
                'Heart_Disease': 0.0, 'Smoker': 0.0
            }
            st.rerun()

    current_data = st.session_state['patient_inputs']

    # 1. DATA COLLECTION FORM
    st.markdown("### 1. Patient Clinical Measurements")
    
    with st.form("clinical_assessment_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<p style='font-weight: 700; color: #0f172a; font-size: 1.05rem;'>Demographics</p>", unsafe_allow_html=True)
            age_in = st.number_input(
                "Age (years)",
                min_value=18.0, max_value=120.0,
                value=float(current_data['Age']),
                step=1.0,
                help="Participant chronological age in years."
            )
            sex_choice = st.selectbox(
                "Biological Sex",
                options=["Male", "Female"],
                index=1 if current_data['Sex'] == 1.0 else 0
            )
            sex_in = 1.0 if sex_choice == "Female" else 0.0

        with col2:
            st.markdown("<p style='font-weight: 700; color: #0f172a; font-size: 1.05rem;'>Vital Measurements</p>", unsafe_allow_html=True)
            sbp_in = st.number_input(
                "Systolic BP (mmHg)",
                min_value=60.0, max_value=260.0,
                value=float(current_data['Systolic_BP']),
                step=1.0,
                help="Oscillometric resting systolic blood pressure."
            )
            dbp_in = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=30.0, max_value=160.0,
                value=float(current_data['Diastolic_BP']),
                step=1.0,
                help="Oscillometric resting diastolic blood pressure."
            )
            bmi_in = st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=10.0, max_value=80.0,
                value=float(current_data['BMI']),
                step=0.1
            )

        with col3:
            st.markdown("<p style='font-weight: 700; color: #0f172a; font-size: 1.05rem;'>Medical History & Laboratory</p>", unsafe_allow_html=True)
            htn_choice = st.selectbox(
                "Hypertension Diagnosis",
                options=["No", "Yes"],
                index=int(current_data['Hypertension'])
            )
            htn_in = 1.0 if htn_choice == "Yes" else 0.0

            hd_choice = st.selectbox(
                "Heart Disease History",
                options=["No", "Yes"],
                index=int(current_data['Heart_Disease']),
                help="History of heart attack, heart failure, coronary disease, or angina."
            )
            hd_in = 1.0 if hd_choice == "Yes" else 0.0

            diab_choice = st.selectbox(
                "Diabetes Mellitus",
                options=["No", "Yes"],
                index=int(current_data['Diabetes'])
            )
            diab_in = 1.0 if diab_choice == "Yes" else 0.0

            smk_choice = st.selectbox(
                "Lifetime Smoking (≥100 cigarettes)",
                options=["No", "Yes"],
                index=int(current_data['Smoker'])
            )
            smk_in = 1.0 if smk_choice == "Yes" else 0.0

            hba1c_in = st.number_input(
                "Glycohemoglobin (HbA1c, %)",
                min_value=3.5, max_value=20.0,
                value=float(current_data['HbA1c']),
                step=0.1
            )

        submit_btn = st.form_submit_button("Run HEAL Assessment", type="primary", use_container_width=True)

    # Input Validation
    val_errors = []
    if sbp_in <= dbp_in:
        val_errors.append(f"Systolic blood pressure ({sbp_in:.0f} mmHg) must be greater than diastolic blood pressure ({dbp_in:.0f} mmHg).")
    if sbp_in < 70 or sbp_in > 250:
        val_errors.append(f"Systolic blood pressure ({sbp_in:.0f} mmHg) is outside typical physiologic limits (70–250 mmHg).")
    if dbp_in < 40 or dbp_in > 150:
        val_errors.append(f"Diastolic blood pressure ({dbp_in:.0f} mmHg) is outside typical physiologic limits (40–150 mmHg).")

    if val_errors:
        for err in val_errors:
            st.error(f"Validation Issue: {err}")
        st.stop()

    # Save inputs
    active_patient = {
        'Systolic_BP': sbp_in,
        'Diastolic_BP': dbp_in,
        'BMI': bmi_in,
        'HbA1c': hba1c_in,
        'Age': age_in,
        'Diabetes': diab_in,
        'Smoker': smk_in,
        'Sex': sex_in,
        'Hypertension': htn_in,
        'Heart_Disease': hd_in
    }
    st.session_state['patient_inputs'] = active_patient

    # -------------------------------------------------------------------------
    # 2. PREDICTION OUTPUT (SAME SCROLL)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. Model Assessment Output")
    
    prob, pred_class, transformed_df = predict_patient(active_patient)
    base_val, shap_df = compute_local_shap(transformed_df)
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown(f"""
        <div class="score-tile">
            <div class="score-label">Estimated Association Score</div>
            <div class="score-value">{prob * 100:.1f}%</div>
            <p style="color: #0f172a; font-size: 1.05rem; font-weight: 700; margin-top: 10px; margin-bottom: 0;">
                Model-estimated probability of reported stroke history
            </p>
            <div style="font-size: 0.92rem; color: #475569; font-weight: 600; margin-top: 8px;">
                Operating Decision Threshold: <strong style="color: #0f172a;">0.50 (50.0%)</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        if pred_class == 1:
            badge_html = '<span class="badge-elevated">Higher model score for reported stroke history</span>'
            explanation = "The patient's combined age, blood pressure, and comorbidity profile placed the estimated probability above the 0.50 operating threshold."
        else:
            badge_html = '<span class="badge-standard">Lower model score for reported stroke history</span>'
            explanation = "The patient's combined profile placed the estimated probability below the 0.50 operating threshold."
            
        st.markdown(f"""
        <div class="score-tile">
            <div class="score-label">Model Classification</div>
            <div style="margin-top: 8px; margin-bottom: 14px;">{badge_html}</div>
            <p style="color: #0f172a; font-size: 1.0rem; font-weight: 600; line-height: 1.5; margin-bottom: 0;">
                Model classification for reported stroke history: <span style="font-weight: 500; color: #334155;">{explanation}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. CONTRIBUTING FACTORS & LOCAL SHAP GRAPH (SAME SCROLL)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. Contributing Factors & Feature Attribution")
    st.markdown("<p style='font-size: 1.0rem; color: #334155; font-weight: 500;'>Visual breakdown of how each clinical measurement pushed the assessment score higher or lower relative to population average.</p>", unsafe_allow_html=True)
    
    graph_col, factor_col = st.columns([3, 2])
    
    with graph_col:
        # High contrast plot styling
        plot_data = shap_df.copy()
        plot_data['Display Name'] = plot_data['Feature'].map(FEATURE_LABELS)
        plot_data = plot_data.sort_values(by='SHAP Value', ascending=True)
        
        bar_colors = ['#dc2626' if val > 0 else '#2563eb' for val in plot_data['SHAP Value']]
        
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        
        bars = ax.barh(plot_data['Display Name'], plot_data['SHAP Value'], color=bar_colors, height=0.55)
        ax.axvline(0, color='#475569', linestyle='--', linewidth=1.2)
        ax.set_xlabel("Impact on Model Log-Odds (Pushing Higher or Lower)", fontsize=11, fontweight='bold', color='#0f172a', labelpad=8)
        ax.set_title("Factors Influencing This Assessment", fontsize=12, fontweight='bold', color='#0f172a', pad=12)
        
        # High-contrast tick labels
        ax.tick_params(axis='x', colors='#0f172a', labelsize=10.5)
        ax.tick_params(axis='y', colors='#0f172a', labelsize=10.5)
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')
            spine.set_linewidth(1.0)
            
        for bar in bars:
            w = bar.get_width()
            align = 'left' if w >= 0 else 'right'
            offset = 0.03 if w >= 0 else -0.03
            ax.annotate(
                f"{w:+.2f}",
                (w + offset, bar.get_y() + bar.get_height() / 2),
                ha=align, va='center', fontsize=9.5, fontweight='bold',
                color='#7f1d1d' if w > 0 else '#1e3a8a'
            )
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with factor_col:
        st.markdown("<p style='font-size: 1.05rem; font-weight: 700; color: #0f172a;'>Factors Increasing Model Score:</p>", unsafe_allow_html=True)
        pos_items = shap_df[shap_df['SHAP Value'] > 0.01].sort_values(by='SHAP Value', ascending=False)
        if len(pos_items) > 0:
            for _, r in pos_items.iterrows():
                fname = FEATURE_LABELS.get(r['Feature'], r['Feature'])
                st.markdown(f'<div class="factor-item-pos">{fname} (+{r["SHAP Value"]:.2f} log-odds)</div>', unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #64748b; font-style: italic;'>None identified for this case.</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.05rem; font-weight: 700; color: #0f172a;'>Factors Decreasing Model Score:</p>", unsafe_allow_html=True)
        neg_items = shap_df[shap_df['SHAP Value'] < -0.01].sort_values(by='SHAP Value', ascending=True)
        if len(neg_items) > 0:
            for _, r in neg_items.iterrows():
                fname = FEATURE_LABELS.get(r['Feature'], r['Feature'])
                st.markdown(f'<div class="factor-item-neg">{fname} ({r["SHAP Value"]:.2f} log-odds)</div>', unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #64748b; font-style: italic;'>None identified for this case.</p>", unsafe_allow_html=True)

    st.markdown("<p style='color: #475569; font-size: 0.92rem; font-weight: 500; margin-top: 10px;'>Note: These values describe statistical feature attributions within the model, not medical causes.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 2: MODEL DETAILS & SCIENTIFIC VALIDATION (SEPARATE TECHNICAL VIEW)
# -----------------------------------------------------------------------------
elif view_mode == "Model Details & Scientific Validation":
    st.markdown("---")
    st.markdown("### Model Details & Scientific Validation")
    st.markdown("<p style='font-size: 1.05rem; color: #334155; font-weight: 500;'>Technical documentation, population-level feature importance, and empirical validation metrics for review.</p>", unsafe_allow_html=True)
    
    tab_global, tab_metrics, tab_method = st.tabs([
        "Global Feature Importance",
        "Performance & Generalization",
        "Methodology & Limitations"
    ])
    
    # Tab 1: Global SHAP
    with tab_global:
        st.markdown("#### Global Feature Importance Ranking (Phase 10 Benchmark)")
        st.markdown("<p style='font-size: 1.0rem; color: #334155;'>Mean absolute attribution across the complete NHANES cohort (N = 6,053).</p>", unsafe_allow_html=True)
        
        if df_shap_global is not None:
            g_col1, g_col2 = st.columns([3, 2])
            
            with g_col1:
                sorted_global = df_shap_global.sort_values(by='Mean Absolute SHAP', ascending=True)
                sorted_global['Display'] = sorted_global['Feature'].map(FEATURE_LABELS)
                
                fig_g, ax_g = plt.subplots(figsize=(8, 5.2))
                fig_g.patch.set_facecolor('#ffffff')
                ax_g.set_facecolor('#ffffff')
                
                bars_g = ax_g.barh(sorted_global['Display'], sorted_global['Mean Absolute SHAP'], color='#0284c7', height=0.55)
                ax_g.set_xlabel("Mean Absolute SHAP Value", fontsize=11, fontweight='bold', color='#0f172a')
                ax_g.set_title("Global Feature Importance (Phase 10)", fontsize=12, fontweight='bold', color='#0f172a', pad=10)
                ax_g.tick_params(axis='x', colors='#0f172a', labelsize=10)
                ax_g.tick_params(axis='y', colors='#0f172a', labelsize=10)
                
                for bar in bars_g:
                    w = bar.get_width()
                    ax_g.annotate(f"{w:.4f}", (w + 0.015, bar.get_y() + bar.get_height() / 2),
                                  ha='left', va='center', fontsize=9.5, fontweight='bold', color='#0f172a')
                    
                plt.tight_layout()
                st.pyplot(fig_g)
                plt.close(fig_g)
                
            with g_col2:
                st.markdown("<p style='font-size: 1.1rem; font-weight: 700; color: #0f172a;'>Key Population Insights:</p>", unsafe_allow_html=True)
                st.markdown(r"""
                <ol style="font-size: 0.98rem; color: #1e293b; line-height: 1.6; font-weight: 500;">
                    <li><strong>Age:</strong> Primary driver of model predictions across the adult population.</li>
                    <li><strong>Smoking History:</strong> Second most influential contributor (≥100 lifetime cigarettes).</li>
                    <li><strong>Hypertension:</strong> Strongest individual diagnosed comorbidity factor.</li>
                    <li><strong>Diastolic BP:</strong> Stronger positive association with reported stroke history than systolic BP once conditioned on hypertension diagnosis.</li>
                    <li><strong>Heart Disease:</strong> Composite cardiovascular indicator with the highest single odds multiplier (3.06×).</li>
                </ol>
                """, unsafe_allow_html=True)
                
            st.markdown("##### Detailed Feature Attribution Inventory")
            st.dataframe(df_shap_global, use_container_width=True)

    # Tab 2: Performance
    with tab_metrics:
        st.markdown("#### Performance Comparison: Holdout vs Generalization")
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Holdout Sensitivity", "77.78%", "Screening Priority")
        m_c2.metric("Holdout Balanced Acc", "75.47%", "Macro-Average")
        m_c3.metric("Holdout ROC-AUC", "0.8259", "Global Discrimination")
        m_c4.metric("Holdout Overall Acc", "73.37%", "Overall Correct")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if evaluation_curves and 'holdout' in evaluation_curves and 'test' in evaluation_curves:
            ho_c = evaluation_curves['holdout']
            te_c = evaluation_curves['test']
            
            c_roc, c_pr = st.columns(2)
            with c_roc:
                fig_r, ax_r = plt.subplots(figsize=(6, 4.4))
                fig_r.patch.set_facecolor('#ffffff')
                ax_r.set_facecolor('#ffffff')
                ax_r.plot(ho_c['fpr'], ho_c['tpr'], label=f"Holdout (AUC = {ho_c['roc_auc']:.4f})", color='#0284c7', linewidth=2.4)
                ax_r.plot(te_c['fpr'], te_c['tpr'], label=f"Generalization (AUC = {te_c['roc_auc']:.4f})", color='#f59e0b', linestyle='--', linewidth=2.0)
                ax_r.plot([0, 1], [0, 1], 'k:', label='Chance Baseline')
                ax_r.set_title("ROC Curve Comparison", fontsize=11, fontweight='bold', color='#0f172a')
                ax_r.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight='bold', color='#0f172a')
                ax_r.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight='bold', color='#0f172a')
                ax_r.tick_params(colors='#0f172a')
                ax_r.legend(loc='lower right', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_r)
                plt.close(fig_r)
                
            with c_pr:
                fig_p, ax_p = plt.subplots(figsize=(6, 4.4))
                fig_p.patch.set_facecolor('#ffffff')
                ax_p.set_facecolor('#ffffff')
                ax_p.plot(ho_c['recall'], ho_c['precision'], label=f"Holdout (PR-AUC = {ho_c['pr_auc']:.4f})", color='#0284c7', linewidth=2.4)
                ax_p.plot(te_c['recall'], te_c['precision'], label=f"Generalization (PR-AUC = {te_c['pr_auc']:.4f})", color='#f59e0b', linestyle='--', linewidth=2.0)
                ax_p.axhline(0.0463, color='#dc2626', linestyle=':', label='Prevalence (4.63%)')
                ax_p.set_title("Precision-Recall Curve Comparison", fontsize=11, fontweight='bold', color='#0f172a')
                ax_p.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight='bold', color='#0f172a')
                ax_p.set_ylabel("Precision (PPV)", fontsize=10, fontweight='bold', color='#0f172a')
                ax_p.tick_params(colors='#0f172a')
                ax_p.legend(loc='upper right', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)

        # Confusion matrix
        st.markdown("##### Confusion Matrices (Threshold = 0.50)")
        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            cm_8b = np.array([[676, 248], [10, 35]])
            fig_cm1, ax_cm1 = plt.subplots(figsize=(4, 3))
            fig_cm1.patch.set_facecolor('#ffffff')
            sns.heatmap(cm_8b, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm1,
                        xticklabels=['Pred No', 'Pred Yes'], yticklabels=['Act No', 'Act Yes'],
                        annot_kws={"fontsize": 11, "fontweight": "bold"})
            ax_cm1.set_title("Phase 8B Holdout (N=969)", fontsize=10.5, fontweight='bold', color='#0f172a')
            ax_cm1.tick_params(colors='#0f172a')
            st.pyplot(fig_cm1)
            plt.close(fig_cm1)
            st.caption("TP=35, FP=248, TN=676, FN=10 (Sensitivity: 77.8%, Specificity: 73.2%)")

        with cm_c2:
            cm_p9 = np.array([[787, 368], [15, 41]])
            fig_cm2, ax_cm2 = plt.subplots(figsize=(4, 3))
            fig_cm2.patch.set_facecolor('#ffffff')
            sns.heatmap(cm_p9, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax_cm2,
                        xticklabels=['Pred No', 'Pred Yes'], yticklabels=['Act No', 'Act Yes'],
                        annot_kws={"fontsize": 11, "fontweight": "bold"})
            ax_cm2.set_title("Phase 9 Generalization (N=1,211)", fontsize=10.5, fontweight='bold', color='#0f172a')
            ax_cm2.tick_params(colors='#0f172a')
            st.pyplot(fig_cm2)
            plt.close(fig_cm2)
            st.caption("TP=41, FP=368, TN=787, FN=15 (Sensitivity: 73.2%, Specificity: 68.1%)")

        if df_metrics_p9 is not None:
            st.markdown("##### Side-by-Side Performance Inventory")
            st.dataframe(df_metrics_p9, use_container_width=True)

    # Tab 3: Methodology
    with tab_method:
        st.markdown("#### Scientific Methodology & Study Limitations")
        st.markdown(r"""
        <div style="font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
        <p><strong>1. Study Population & Cohort Construction:</strong><br>
        • Dataset: CDC National Health and Nutrition Examination Survey (NHANES) 2021–2023 cycle.<br>
        • Sample size: <em>N = 6,053</em> non-institutionalized US adults aged 18 to 80 with complete examinations and laboratory assays.<br>
        • Target Phenotype: Doctor-diagnosed history of stroke derived from questionnaire instrument <code>MCQ160F</code> (<em>N=280</em> positive, 4.63% prevalence). Non-responses were cleanly excluded.</p>

        <p><strong>2. Model Formulation & Class Imbalance:</strong><br>
        • Model: Logistic Regression (<code>C=1.0</code>, L2 regularization, <code>class_weight='balanced'</code>).<br>
        • Cost-sensitive weighting: Scaled minority class loss inversely to class frequency (20.6× higher penalty for false negatives).<br>
        • Operating threshold: Systematically locked at <strong>&tau; = 0.50</strong> on cross-validation to maximize balanced accuracy subject to screening sensitivity &ge; 75%.</p>

        <p><strong>3. Critical Epidemiological Limitations:</strong><br>
        • <strong>Cross-Sectional Nature:</strong> NHANES measures stroke history and clinical biomarkers concurrently. This model assesses statistical associations with <em>past reported stroke history</em>, NOT longitudinal prospective future stroke.<br>
        • <strong>Self-Reported Outcome:</strong> Stroke history is self-reported, not verified through hospital discharge records or neuroimaging.<br>
        • <strong>Survivorship Bias:</strong> The cohort includes community-dwelling stroke survivors; fatal or institutionalized strokes are not represented.<br>
        • <strong>Non-Causal SHAP:</strong> Feature attributions describe statistical model weights, not biological or etiologic causation.</p>
        </div>
        """, unsafe_allow_html=True)
