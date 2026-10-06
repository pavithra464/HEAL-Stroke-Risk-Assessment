# HEAL: Hybrid Explainable Adaptive Learning for Stroke Risk Assessment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Clinical Decision-Support Research Prototype**  
> *Benchmark Model: Locked Phase 8B Logistic Regression Pipeline ($\tau = 0.50$)*

---

## 📋 Overview
HEAL is a clinical decision-support research prototype designed to assess statistical associations between patient clinical measurements and self-reported stroke history based on CDC NHANES (2021–2023). It incorporates model governance, transparent risk estimation, and mathematical exactness in local SHAP feature attribution.

> **IMPORTANT CLINICAL & REGULATORY DISCLAIMER**  
> **Research Prototype Only:** HEAL does not provide medical diagnoses, does not forecast future stroke incidence, and does not replace the professional clinical judgment of licensed medical providers.

---

## 🚀 Deployment Guide: Streamlit Community Cloud

Deploying HEAL to **Streamlit Community Cloud** takes under 2 minutes:

### Step 1: Create a GitHub Repository
1. Go to [GitHub New Repository](https://github.com/new).
2. Name your repository (e.g., `HEAL-Stroke-Risk-Assessment` or `HEAL_Project`).
3. Set visibility to **Public** (or Private if you have linked private repos to Streamlit Cloud).
4. Run the following commands in your local project terminal:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Select your repository: `<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>`.
4. Set the branch: `main`.
5. Set **Main file path**: `HEAL_Streamlit/app.py` (or `app.py`).
6. Click **Deploy!** 🎈

Streamlit Cloud will install dependencies from `requirements.txt` and launch your live public URL within ~1–2 minutes.

---

## 💻 Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run HEAL_Streamlit/app.py
```
Or simply:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Automated Integration Tests
```bash
python3 HEAL_Streamlit/test_app.py
```

---

## 📁 Repository Structure
```
HEAL_Project/
├── app.py                     # Root entry-point delegator
├── requirements.txt           # Deployment dependency specifications
├── .gitignore                 # Standard Python/data ignore rules
├── README.md                  # Project & deployment guide
│
├── HEAL_Streamlit/            # Standalone Streamlit Application
│   ├── app.py                 # Multi-page interactive application
│   ├── test_app.py            # 6-point automated verification suite
│   ├── requirements.txt       # Subfolder dependency manifest
│   ├── .streamlit/config.toml # Theme and layout configuration
│   ├── model/                 # Locked Phase 8B Model & SHAP artifacts
│   │   ├── final_model_pipeline_phase8b.joblib
│   │   ├── final_features_phase8b.json
│   │   ├── final_threshold_phase8b.json
│   │   └── shap_background.json
│   └── assets/                # Validation metrics and evaluation curves
│       ├── final_metrics_phase8b.csv
│       ├── phase9_generalization_metrics.csv
│       ├── shap_global_feature_importance_phase10.csv
│       ├── shap_representative_cases_phase10.csv
│       └── evaluation_curves.json
│
└── 01-11 Notebooks/           # NHANES Research & Development Pipeline
```
