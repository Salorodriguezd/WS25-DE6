import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


TARGET_COL = "Late_delivery_risk"


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple feature selection:
     - Drop the target column and obvious ID columns
     - Keep only numerical features for this baseline version
       (can be extended later with selected categorical features + one-hot encoding)
    """

    df = df.copy()

    drop_cols = [
        TARGET_COL,
        "Order Id",
        "Order Item Id",
        "Order Item Cardprod Id",
        "Customer Id",
        "Order Customer Id",
        "Product Card Id",
        "Product Category Id",
        "Order Zipcode",
        "Customer Zipcode",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    # For now, use only numerical features (simple baseline)
    X_num = X.select_dtypes(include=[np.number])

    return X_num


def _compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_models_dataco(
    dataco_path: str = "data/DataCo_clean_dates_ddmmyyyy.csv",
    output_table_path: str = "tables/RQ1_Table1_model_performance.csv",
) -> pd.DataFrame:

    """
    Train several classification models (Logistic Regression, Random Forest)
    on the DataCo dataset and generate a table of performance metrics.
    """

    print(f"Loading DataCo from: {dataco_path}")
    df = pd.read_csv(dataco_path, low_memory=False)
    print("DataCo shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataCo dataset.")

    # Target
    y = df[TARGET_COL].astype(int)

    # Features
    X = _select_features(df)
    print("Feature matrix shape:", X.shape)

    # Train/validation split (simple random split; can be replaced with time-based split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # ----------------------------------------------------------
    # 1) Logistic Regression (baseline)
    # ----------------------------------------------------------
    log_reg = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
    )
    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_val)
    y_proba_lr = log_reg.predict_proba(X_val)[:, 1]

    metrics_lr = _compute_metrics(y_val, y_pred_lr, y_proba_lr)
    metrics_lr["model"] = "Logistic Regression"
    results.append(metrics_lr)

    # ----------------------------------------------------------
    # 2) Random Forest
    # ----------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_val)
    y_proba_rf = rf.predict_proba(X_val)[:, 1]

    metrics_rf = _compute_metrics(y_val, y_pred_rf, y_proba_rf)
    metrics_rf["model"] = "Random Forest"
    results.append(metrics_rf)

    # TODO: later add XGBoost / LightGBM

    # ----------------------------------------------------------
    # 3) Aggregate and save results table
    # ----------------------------------------------------------
    df_results = pd.DataFrame(results)[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]

    print("Model performance table:")
    print(df_results.to_markdown(index=False))

    output_table_path = Path(output_table_path)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_table_path, index=False)
    print(f"Saved RQ1 performance table to: {output_table_path}")

    return df_results


if __name__ == "__main__":
    # From the repo root, run:
    #   python -m src.modeling.train_models_dataco
    default_dataco = os.environ.get(
        "DE_DATACO_PATH", "data/DataCo_clean_dates_ddmmyyyy.csv"
    )
    default_output = os.environ.get(
        "DE_RQ1_TABLE_PATH", "tables/RQ1_Table1_model_performance.csv"
    )

    train_models_dataco(
        dataco_path=default_dataco,
        output_table_path=default_output,
    )


