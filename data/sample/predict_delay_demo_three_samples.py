"""
Delay Prediction Demo
Shows how the trained model predicts delivery delays for new orders.
"""

import pandas as pd
import joblib
from pathlib import Path


def load_model(model_path: str = "models/xgboost_delay_predictor.pkl"):
    """Load the trained model and feature names."""
    saved = joblib.load(model_path)
    return saved['model'], saved['feature_names'], saved['metrics']


def predict_delay(order_data: dict, model, feature_names: list):
    """
    Predict delay for a single order.
    
    Args:
        order_data: Dictionary with feature values (partial features OK)
        model: Trained XGBoost model
        feature_names: List of feature names in correct order
    
    Returns:
        prediction (0/1), probability (float)
    """
    # ✅ Create empty DataFrame with all required features (default = 0)
    X_new = pd.DataFrame([{feat: 0 for feat in feature_names}])
    
    # ✅ Fill in provided features
    for key, value in order_data.items():
        if key in feature_names:
            X_new[key] = value
    
    # Predict
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]
    
    return prediction, probability


def demo():
    """Run prediction demo with sample orders."""
    
    # Load model
    print("Loading trained model...")
    model, feature_names, metrics = load_model()
    print(f"✓ Model loaded (ROC-AUC: {metrics['roc_auc']:.4f})")
    print(f"✓ Using {len(feature_names)} features\n")
    
    # Sample orders
    sample_orders = [
        {
            "name": "High-Risk Order (Short scheduled + High risk route)",
            "Days for shipment (scheduled)": 2,
            "Days for shipping (real)": 1,
            "ship_mode_risk": 0.8,
            "lead_time_days": 3,
            "route_delay_rate": 0.6,
            "customer_delay_history": 0.4,
        },
        {
            "name": "Low-Risk Order (Sufficient time + Safe route)",
            "Days for shipment (scheduled)": 10,
            "Days for shipping (real)": 5,
            "ship_mode_risk": 0.2,
            "lead_time_days": 12,
            "route_delay_rate": 0.1,
            "customer_delay_history": 0.05,
        },
        {
            "name": "Medium-Risk Order",
            "Days for shipment (scheduled)": 5,
            "Days for shipping (real)": 3,
            "ship_mode_risk": 0.5,
            "lead_time_days": 7,
            "route_delay_rate": 0.3,
            "customer_delay_history": 0.2,
        },
    ]
    
    # Run predictions
    for i, order in enumerate(sample_orders, 1):
        print(f"{'='*60}")
        print(f"Example {i}: {order['name']}")
        print(f"{'='*60}")
        
        order_features = {k: v for k, v in order.items() if k != 'name'}
        
        print("\nInput features (key predictors):")
        for key, value in order_features.items():
            print(f"  - {key}: {value}")
        
        prediction, probability = predict_delay(order_features, model, feature_names)
        
        print(f"\nRESULT:")
        if prediction == 1:
            print(f"  ⚠️  DELAY PREDICTED!")
            print(f"  📊 Delay probability: {probability:.1%}")
        else:
            print(f"  ✅ ON TIME")
            print(f"  📊 Delay risk: {probability:.1%}")
        
        print()


if __name__ == "__main__":
    demo()
