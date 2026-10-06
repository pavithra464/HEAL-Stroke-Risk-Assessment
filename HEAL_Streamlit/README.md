# HEAL: Hybrid Explainable Adaptive Learning for Stroke Risk Assessment
## Explainable Machine-Learning Clinical Decision-Support Prototype

### Clinical Decision-Support Research Prototype — Review Version
*Benchmark Model: Locked Phase 8B Logistic Regression Pipeline ($\tau = 0.50$)*

---

### Project Overview
The **HEAL** prototype is a research clinical decision-support system designed to assess the statistical association between patient clinical measurements and self-reported stroke history based on the CDC National Health and Nutrition Examination Survey (NHANES 2021–2023).

> **IMPORTANT CLINICAL & REGULATORY DISCLAIMER**  
> **Research Prototype Only:** HEAL does not provide medical diagnoses, does not forecast future stroke risk, and does not replace the professional clinical judgment of licensed medical providers. SHAP feature attributions reflect statistical associations within the trained model and do NOT establish clinical causality.

---

### Directory Architecture
```
HEAL_Streamlit/
│
├── app.py                     # Main multi-page Streamlit application
├── requirements.txt           # Dependency specifications
├── README.md                  # Project documentation & run guide
│
├── model/                     # Locked Phase 8B Model Artifacts
│   ├── final_model_pipeline_phase8b.joblib
│   ├── final_features_phase8b.json
│   ├── final_threshold_phase8b.json
│   └── shap_background.json  # Precomputed development background attributions
│
└── assets/                    # Empirical Validation & Interpretability Artifacts
    ├── final_metrics_phase8b.csv
    ├── phase9_generalization_metrics.csv
    ├── shap_global_feature_importance_phase10.csv
    ├── shap_representative_cases_phase10.csv
    └── evaluation_curves.json
```

---

### Key Application Features
1. **Home:** Project architecture overview, model governance status, and visible regulatory disclaimers.
2. **Clinical Assessment:** Structured clinical intake interface (Demographics, Vitals, History, Laboratory) with strict physiological range and sanity validation.
3. **Model Result:** Clear presentation of model-estimated probability of reported stroke history and binary classification under the locked $\tau = 0.50$ threshold.
4. **SHAP Explainability:** Case-level local attribution waterfall charts identifying positive and negative feature pushes.
5. **Global Model Explanation:** Empirical Phase 10 global SHAP feature importance rankings and beta coefficients.
6. **Model Performance:** Comparative evaluation dashboard displaying holdout ($N=969$) and Phase 9 generalization ($N=1,211$) metrics, confusion matrices, ROC curves, and PR curves.
7. **About & Methodology:** Full disclosure of data provenance, model architecture, preprocessing pipelines, and epidemiological limitations.

---

### Quickstart Guide

#### 1. Setup Environment
Ensure Python 3.9+ is active:
```bash
cd HEAL_Streamlit
pip install -r requirements.txt
```

#### 2. Launch the Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

### Locked Model Specifications
- **Algorithm:** Logistic Regression ($C=1.0$, L2 Penalty, `class_weight='balanced'`)
- **Operating Threshold:** $\tau = 0.50$ (Locked)
- **Features (10):**
  - Continuous (5): Systolic BP, Diastolic BP, BMI, HbA1c, Age
  - Categorical / Binary (5): Diabetes, Lifetime Smoker, Sex, Hypertension, Heart Disease Composite
- **Validation Metrics (Phase 8B Holdout):**
  - Accuracy: $73.37\%$
  - Sensitivity: $77.78\%$
  - Specificity: $73.16\%$
  - Balanced Accuracy: $75.47\%$
  - ROC-AUC: $0.8259$
  - PR-AUC: $0.1832$
