"""
Delay Prediction Demo
Shows how the trained model predicts delivery delays using REAL validation data.
"""

import pandas as pd
import joblib
from pathlib import Path


def load_model(model_path: str = "models/xgboost_delay_predictor.pkl"):
    """Load the trained model and feature names."""
    saved = joblib.load(model_path)
    return saved['model'], saved['feature_names'], saved['metrics']


def demo_with_real_data():
    """Demo using real validation samples from the dataset."""
    
    # Load model
    print("="*70)
    print("DELAY PREDICTION DEMO - Using Real Data Samples")
    print("="*70)
    print("\nLoading model and validation data...")
    
    model, feature_names, metrics = load_model()
    print(f"✓ Model loaded successfully")
    print(f"✓ Model ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"✓ Using {len(feature_names)} features")
    
    # Load actual data
    df = pd.read_csv("data/merged/merged_with_engineered_features.csv", low_memory=False)
    print(f"✓ Loaded {len(df):,} orders from dataset\n")
    
    # Get interesting samples
    # 1. High-risk delayed order (actual delay)
    delayed_candidates = df[df['Late_delivery_risk'] == 1]
    delayed_sample = delayed_candidates.sample(1, random_state=42)
    
    # 2. Low-risk on-time order (actual on-time)
    ontime_candidates = df[df['Late_delivery_risk'] == 0]
    ontime_sample = ontime_candidates.sample(1, random_state=42)
    
    # 3. Edge case: order that looks risky but was on-time
    edge_case = df[
        (df['Late_delivery_risk'] == 0) & 
        (df['Days for shipment (scheduled)'] < 5)
    ].sample(1, random_state=123) if len(df[
        (df['Late_delivery_risk'] == 0) & 
        (df['Days for shipment (scheduled)'] < 5)
    ]) > 0 else ontime_sample
    
    samples = [
        ("Example 1: ACTUAL DELAYED ORDER", delayed_sample, 1),
        ("Example 2: ACTUAL ON-TIME ORDER", ontime_sample, 0),
        ("Example 3: EDGE CASE (Looks risky but on-time)", edge_case, 0),
    ]
    
    # Run predictions on real samples
    for name, sample_df, actual_label in samples:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        
        # Extract all features for prediction
        X_sample = sample_df[feature_names].fillna(0)
        
        # Show key features that matter most (from RQ3 analysis)
        print("\nKey predictive features (from model):")
        key_features = [
            'Days for shipment (scheduled)',
            'Days for shipping (real)',
            'ship_mode_risk',
            'lead_time_days',
            'route_delay_rate',
            'customer_delay_history',
        ]
        
        for feat in key_features:
            if feat in X_sample.columns:
                value = X_sample[feat].values[0]
                print(f"  • {feat:35s}: {value:.3f}")
        
        # Make prediction
        prediction = model.predict(X_sample)[0]
        probability = model.predict_proba(X_sample)[0][1]
        
        # Show results
        print(f"\n{'─'*70}")
        print("📊 PREDICTION RESULT:")
        print(f"{'─'*70}")
        print(f"  Model Prediction:  {'🔴 DELAY' if prediction == 1 else '🟢 ON TIME'}")
        print(f"  Delay Probability: {probability:.1%}")
        print(f"  Actual Outcome:    {'🔴 DELAYED' if actual_label == 1 else '🟢 ON TIME'}")
        
        # Check if prediction matches reality
        is_correct = (prediction == actual_label)
        print(f"\n  Result: {'✅ CORRECT PREDICTION' if is_correct else '❌ INCORRECT PREDICTION'}")
        
        # Explanation
        if actual_label == 1:  # Was delayed
            if prediction == 1:
                print(f"  → Model successfully identified high delay risk ({probability:.1%})")
            else:
                print(f"  → Model missed this delay (only {probability:.1%} risk predicted)")
        else:  # Was on-time
            if prediction == 0:
                print(f"  → Model correctly predicted low delay risk ({probability:.1%})")
            else:
                print(f"  → Model over-predicted delay risk ({probability:.1%})")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("DEMO SUMMARY")
    print(f"{'='*70}")
    print(f"Model Performance on Full Dataset:")
    print(f"  • ROC-AUC Score:     {metrics['roc_auc']:.4f}")
    print(f"  • Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  • Precision:         {metrics['precision']:.4f}")
    print(f"  • Recall:            {metrics['recall']:.4f}")
    print(f"  • F1-Score:          {metrics['f1']:.4f}")
    print(f"\nThese samples demonstrate real-world prediction capability.")
    print(f"{'='*70}\n")


def demo_with_custom_input():
    """Demo with manually created order scenarios (for demonstration)."""
    
    print("="*70)
    print("DELAY PREDICTION DEMO - Custom Scenarios")
    print("="*70)
    print("\nLoading model...")
    
    model, feature_names, metrics = load_model()
    print(f"✓ Model loaded (ROC-AUC: {metrics['roc_auc']:.4f})")
    print(f"✓ Using {len(feature_names)} features\n")
    
    # Get typical values from training data to make realistic samples
    df = pd.read_csv("data/merged/merged_with_engineered_features.csv", low_memory=False)
    
    # Create realistic complete samples based on actual data patterns
    delayed_sample = df[df['Late_delivery_risk'] == 1].sample(1, random_state=42)
    ontime_sample = df[df['Late_delivery_risk'] == 0].sample(1, random_state=42)
    
    # Modify key features to create scenarios
    scenarios = []
    
    # Scenario 1: High risk (based on delayed order, make it more risky)
    high_risk = delayed_sample[feature_names].copy()
    high_risk['Days for shipment (scheduled)'] = 2  # Very short
    high_risk['ship_mode_risk'] = 0.8  # High risk mode
    high_risk['route_delay_rate'] = 0.7  # High delay rate
    scenarios.append(("HIGH-RISK ORDER (Short time + Risky route)", high_risk))
    
    # Scenario 2: Low risk (based on on-time order, make it safer)
    low_risk = ontime_sample[feature_names].copy()
    low_risk['Days for shipment (scheduled)'] = 10  # Plenty of time
    low_risk['ship_mode_risk'] = 0.1  # Low risk mode
    low_risk['route_delay_rate'] = 0.05  # Low delay rate
    scenarios.append(("LOW-RISK ORDER (Sufficient time + Safe route)", low_risk))
    
    for name, X_sample in scenarios:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        
        print("\nKey features:")
        key_features = [
            'Days for shipment (scheduled)',
            'Days for shipping (real)',
            'ship_mode_risk',
            'route_delay_rate',
        ]
        
        for feat in key_features:
            if feat in X_sample.columns:
                print(f"  • {feat:35s}: {X_sample[feat].values[0]:.3f}")
        
        # Predict
        prediction = model.predict(X_sample)[0]
        probability = model.predict_proba(X_sample)[0][1]
        
        print(f"\n{'─'*70}")
        print("📊 PREDICTION:")
        print(f"{'─'*70}")
        print(f"  Result:      {'🔴 DELAY PREDICTED' if prediction == 1 else '🟢 ON TIME'}")
        print(f"  Probability: {probability:.1%}")


if __name__ == "__main__":
    # Choose which demo to run
    print("\n")
    print("Choose demo mode:")
    print("1. Real data samples (recommended)")
    print("2. Custom scenarios")
    
    choice = input("\nEnter 1 or 2 (or press Enter for default=1): ").strip()
    
    if choice == "2":
        demo_with_custom_input()
    else:
        demo_with_real_data()
