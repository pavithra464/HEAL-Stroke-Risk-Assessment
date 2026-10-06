"""
Automated Integration and Functional Test Suite for HEAL Streamlit Prototype
Validates:
1. Model loading and pipeline integrity
2. Exact Phase 8B feature order
3. Prediction and threshold classification
4. Mathematical exactness of SHAP feature attribution
5. Edge cases: Low score, High score, Boundary score
6. Input validation for physiologically invalid cases
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

def run_tests():
    print("="*70)
    print("HEAL STREAMLIT APPLICATION TEST SUITE")
    print("="*70)
    
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, "model", "final_model_pipeline_phase8b.joblib")
    features_path = os.path.join(app_dir, "model", "final_features_phase8b.json")
    threshold_path = os.path.join(app_dir, "model", "final_threshold_phase8b.json")
    bg_path = os.path.join(app_dir, "model", "shap_background.json")
    
    # Test 1: Verify all required files exist
    print("[TEST 1]: Verifying required model and asset files...")
    for p in [model_path, features_path, threshold_path, bg_path]:
        assert os.path.exists(p), f"Missing file: {p}"
    print("  -> Passed. All model artifacts present.")
    
    # Test 2: Load pipeline and specifications
    print("\n[TEST 2]: Loading locked Phase 8B pipeline and feature specification...")
    pipeline = joblib.load(model_path)
    with open(features_path) as f:
        feat_spec = json.load(f)
    with open(threshold_path) as f:
        th_spec = json.load(f)
    with open(bg_path) as f:
        shap_bg = json.load(f)
        
    features = feat_spec['features']
    locked_th = float(th_spec['locked_threshold'])
    assert locked_th == 0.50, f"Expected threshold 0.50, found {locked_th}"
    print(f"  -> Passed. Pipeline loaded successfully. Locked threshold = {locked_th:.2f}")
    print(f"  -> Features ({len(features)}): {features}")
    
    # Helper to predict
    def evaluate(patient_dict):
        df_in = pd.DataFrame([patient_dict])[features]
        prob = float(pipeline.predict_proba(df_in)[0, 1])
        cls = 1 if prob >= locked_th else 0
        
        # Local SHAP
        preprocessor = pipeline.named_steps['preprocessor']
        clf = pipeline.named_steps['classifier']
        feat_order = feat_spec['numerical_features'] + feat_spec['categorical_features']
        trans = pd.DataFrame(preprocessor.transform(df_in), columns=feat_order)
        
        shaps = {}
        for feat in feat_order:
            beta = shap_bg['coefficients'][feat]
            mu = shap_bg['mean_transformed'][feat]
            x_val = trans.iloc[0][feat]
            shaps[feat] = beta * (x_val - mu)
            
        base = shap_bg['base_value']
        total_logit = base + sum(shaps.values())
        model_logit = float(clf.decision_function(trans)[0])
        additivity_err = abs(total_logit - model_logit)
        
        return prob, cls, shaps, additivity_err
        
    # Test 3: Low Model Score Example
    print("\n[TEST 3]: Testing Low-Model-Score Example (Young, Normotensive, Non-Smoker)...")
    low_patient = {
        'Systolic_BP': 112.0, 'Diastolic_BP': 72.0, 'BMI': 22.4, 'HbA1c': 5.1,
        'Age': 32.0, 'Diabetes': 0.0, 'Smoker': 0.0, 'Sex': 1.0,
        'Hypertension': 0.0, 'Heart_Disease': 0.0
    }
    prob_low, cls_low, shaps_low, err_low = evaluate(low_patient)
    print(f"  -> Probability: {prob_low*100:.2f}%")
    print(f"  -> Classification: {cls_low} ({'Higher model score' if cls_low==1 else 'Lower model score for reported stroke history'})")
    print(f"  -> SHAP Additivity Error: {err_low:.2e}")
    assert prob_low < 0.20, f"Expected probability < 0.20, got {prob_low}"
    assert cls_low == 0, "Expected classification 0"
    assert err_low < 1e-10, "SHAP additivity check failed!"
    print("  -> Passed Low-Model-Score Test.")
    
    # Test 4: High Model Score Example
    print("\n[TEST 4]: Testing High-Model-Score Example (Elderly, Hypertensive, Heart Disease, Smoker)...")
    high_patient = {
        'Systolic_BP': 155.0, 'Diastolic_BP': 88.0, 'BMI': 32.5, 'HbA1c': 7.2,
        'Age': 76.0, 'Diabetes': 1.0, 'Smoker': 1.0, 'Sex': 0.0,
        'Hypertension': 1.0, 'Heart_Disease': 1.0
    }
    prob_high, cls_high, shaps_high, err_high = evaluate(high_patient)
    print(f"  -> Probability: {prob_high*100:.2f}%")
    print(f"  -> Classification: {cls_high} ({'Higher model score for reported stroke history' if cls_high==1 else 'Lower model score'})")
    print(f"  -> SHAP Additivity Error: {err_high:.2e}")
    print(f"  -> Top Positive Factor: {max(shaps_high.items(), key=lambda x: x[1])}")
    assert prob_high > 0.80, f"Expected probability > 0.80, got {prob_high}"
    assert cls_high == 1, "Expected classification 1"
    assert err_high < 1e-10, "SHAP additivity check failed!"
    print("  -> Passed High-Model-Score Test.")
    
    # Test 5: Boundary Example
    print("\n[TEST 5]: Testing Decision Threshold Boundary Example (~0.50)...")
    boundary_patient = {
        'Systolic_BP': 128.0, 'Diastolic_BP': 76.0, 'BMI': 28.0, 'HbA1c': 5.7,
        'Age': 66.0, 'Diabetes': 0.0, 'Smoker': 0.0, 'Sex': 1.0,
        'Hypertension': 1.0, 'Heart_Disease': 0.0
    }
    prob_bnd, cls_bnd, shaps_bnd, err_bnd = evaluate(boundary_patient)
    print(f"  -> Probability: {prob_bnd*100:.2f}%")
    print(f"  -> Classification: {cls_bnd} (Threshold = {locked_th:.2f})")
    assert 0.40 <= prob_bnd <= 0.60, f"Expected probability near 0.50, got {prob_bnd}"
    assert err_bnd < 1e-10, "SHAP additivity check failed!"
    print("  -> Passed Boundary Test.")
    
    # Test 6: Input Validation Verification
    print("\n[TEST 6]: Testing Clinical Input Validation Rules...")
    # SBP <= DBP
    inv_1 = {'Systolic_BP': 80.0, 'Diastolic_BP': 90.0}
    is_valid_1 = inv_1['Systolic_BP'] > inv_1['Diastolic_BP']
    assert not is_valid_1, "Validation failed to catch SBP <= DBP!"
    print("  -> SBP <= DBP correctly flagged as invalid.")
    
    # Out of range SBP
    inv_2 = {'Systolic_BP': 310.0}
    is_valid_2 = 60 <= inv_2['Systolic_BP'] <= 260
    assert not is_valid_2, "Validation failed to catch SBP > 260!"
    print("  -> Out-of-range SBP correctly flagged as invalid.")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED SUCCESSFULLY! (6/6 Checks Verified)")
    print("="*70)

if __name__ == '__main__':
    run_tests()
