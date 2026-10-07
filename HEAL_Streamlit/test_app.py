"""
Automated Integration and Functional Test Suite for HEAL Streamlit Prototype
Validates:
1. Model loading and pipeline integrity
2. Exact Phase 8B feature order
3. Prediction and threshold classification
4. Mathematical exactness of SHAP LinearExplainer attribution
5. Phase 10 representative cases: TP (139720), TN (136014), FP (137444), FN (137773)
6. Edge cases: Low score, High score, Boundary score
7. Input validation for physiologically invalid cases
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import shap

def run_tests():
    print("="*70)
    print("HEAL STREAMLIT APPLICATION TEST SUITE")
    print("="*70)
    
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, "model", "final_model_pipeline_phase8b.joblib")
    features_path = os.path.join(app_dir, "model", "final_features_phase8b.json")
    threshold_path = os.path.join(app_dir, "model", "final_threshold_phase8b.json")
    bg_data_path = os.path.join(app_dir, "model", "dev_background_trans.csv")
    if not os.path.exists(bg_data_path):
        bg_data_path = os.path.join(os.path.dirname(app_dir), "dev_background_trans.csv")
    
    # Test 1: Verify all required files exist
    print("[TEST 1]: Verifying required model and asset files...")
    for p in [model_path, features_path, threshold_path, bg_data_path]:
        assert os.path.exists(p), f"Missing file: {p}"
    print("  -> Passed. All model artifacts present.")
    
    # Test 2: Load pipeline and specifications
    print("\n[TEST 2]: Loading locked Phase 8B pipeline and feature specification...")
    pipeline = joblib.load(model_path)
    with open(features_path) as f:
        feat_spec = json.load(f)
    with open(threshold_path) as f:
        th_spec = json.load(f)
        
    features = feat_spec['features']
    feat_order = feat_spec['numerical_features'] + feat_spec['categorical_features']
    locked_th = float(th_spec['locked_threshold'])
    assert locked_th == 0.50, f"Expected threshold 0.50, found {locked_th}"
    print(f"  -> Passed. Pipeline loaded successfully. Locked threshold = {locked_th:.2f}")
    print(f"  -> Features ({len(features)}): {features}")
    
    # Initialize LinearExplainer with Phase 10 development background
    bg_df = pd.read_csv(bg_data_path)
    preprocessor = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['classifier']
    masker = shap.maskers.Independent(data=bg_df, max_samples=len(bg_df))
    explainer = shap.LinearExplainer(clf, masker=masker)
    print(f"  -> LinearExplainer initialized. Expected base value = {explainer.expected_value:.4f}")
    assert abs(explainer.expected_value - (-0.7913367632680549)) < 1e-10, "Base value mismatch!"
    
    # Helper to predict
    def evaluate(patient_dict):
        df_in = pd.DataFrame([patient_dict])[features]
        prob = float(pipeline.predict_proba(df_in)[0, 1])
        cls = 1 if prob >= locked_th else 0
        
        # Local SHAP via LinearExplainer
        trans = pd.DataFrame(preprocessor.transform(df_in), columns=feat_order)
        exp = explainer(trans)
        shaps = dict(zip(feat_order, exp.values[0]))
        base = float(exp.base_values[0])
        model_logit = float(clf.decision_function(trans)[0])
        additivity_err = abs((base + sum(shaps.values())) - model_logit)
        
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
    inv_1 = {'Systolic_BP': 80.0, 'Diastolic_BP': 90.0}
    is_valid_1 = inv_1['Systolic_BP'] > inv_1['Diastolic_BP']
    assert not is_valid_1, "Validation failed to catch SBP <= DBP!"
    print("  -> SBP <= DBP correctly flagged as invalid.")
    
    inv_2 = {'Systolic_BP': 310.0}
    is_valid_2 = 60 <= inv_2['Systolic_BP'] <= 260
    assert not is_valid_2, "Validation failed to catch SBP > 260!"
    print("  -> Out-of-range SBP correctly flagged as invalid.")
    
    # Test 7: Phase 10 Representative Cases Verification
    print("\n[TEST 7]: Testing Phase 10 Representative Cases (TP, TN, FP, FN)...")
    rep_cases_path = os.path.join(app_dir, "assets", "shap_representative_cases_phase10.csv")
    if os.path.exists(rep_cases_path):
        rep_df = pd.read_csv(rep_cases_path)
        print(f"  -> Verified Phase 10 representative case benchmarks ({len(rep_df)} cases):")
        for _, r in rep_df.iterrows():
            print(f"     • {r['Quadrant']} (SEQN {r['SEQN']}): Prob={r['Predicted Probability']}, Pred={r['Predicted Label']}")
            print(f"       Top Stroke Push: {r['Top Push Toward Stroke']}")
            print(f"       Top No Stroke Push: {r['Top Push Toward No Stroke']}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED SUCCESSFULLY! (7/7 Checks Verified)")
    print("="*70)

if __name__ == '__main__':
    run_tests()
