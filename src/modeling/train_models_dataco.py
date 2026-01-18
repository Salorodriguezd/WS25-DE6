"""
Modeling for RQ1: baseline models, robustness, and feature importance.

Outputs (saved under ./tables and ./figures):

- tables/RQ1_Table1_model_performance.csv
    Comparative performance of classical models on the (merged) DataCo dataset.

- tables/RQ1_Table2_time_based_performance.csv
    Time-based validation performance across multiple train/test splits.

- figures/RQ1_Fig1_feature_importance.pdf
    Horizontal bar chart of XGBoost feature importances.

- figures/RQ1_Fig2_time_based_performance.pdf
    Line plot showing ROC-AUC over time-based splits using XGBoost.
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET_COL = "Late_delivery_risk"
DATE_COL = "order date (DateOrders)"  # used for time-based splits


# ---------------------------------------------------------------------
# Utility: feature selection and metrics
# ---------------------------------------------------------------------


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline feature selection for RQ1 Table 1 / Figure 1.

    - Drop target and obvious ID columns.
    - Keep only numerical features.
    - Drop columns that are entirely NaN (e.g., shipment_weight if empty).
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

    # numeric-only baseline
    X_num = X.select_dtypes(include=[np.number])

    # drop columns that are entirely NaN (shipment_weight, avg_weight, etc.)
    X_num = X_num.loc[:, ~X_num.isna().all()]

    return X_num


def _select_features_for_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature selection for time-based validation.

    Similar to _select_features, but:
    - Also drops the date column.
    - If no numeric columns remain, attempts to coerce object columns to numeric.
    """
    df = df.copy()

    drop_cols = [
        TARGET_COL,
        DATE_COL,
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

    # 1) numeric columns first
    X_num = X.select_dtypes(include=[np.number])

    # 2) if no numeric columns, try coercing object columns to numeric
    if X_num.shape[1] == 0:
        obj_cols = X.select_dtypes(include=["object"]).columns
        converted = {}
        for c in obj_cols:
            col_conv = pd.to_numeric(X[c], errors="coerce")
            if col_conv.notna().sum() > 0:
                converted[c] = col_conv
        if converted:
            X_num = pd.DataFrame(converted, index=X.index)

    if X_num.shape[1] == 0:
        raise ValueError(
            "No usable features for time-based validation after conversion."
        )

    # drop columns that are entirely NaN
    X_num = X_num.loc[:, ~X_num.isna().all()]

    return X_num


def _compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ---------------------------------------------------------------------
# RQ1 – Table 1: baseline model comparison
# ---------------------------------------------------------------------


def run_baseline_models(
    df: pd.DataFrame,
    output_table_path: str = "tables/RQ1_Table1_model_performance.csv",
) -> pd.DataFrame:
    """
    Train Logistic Regression, Random Forest, and XGBoost on a random
    train/validation split and save a comparison table of performance metrics.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = df[TARGET_COL].astype(int)

    # feature DataFrame (before imputation) for names
    X_df = _select_features(df)
    feature_names = X_df.columns.tolist()

    # simple mean-imputation for NaNs
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X_df)

    print("Feature matrix shape:", X.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # Logistic Regression
    log_reg = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_val)
    y_proba_lr = log_reg.predict_proba(X_val)[:, 1]

    metrics_lr = _compute_metrics(y_val, y_pred_lr, y_proba_lr)
    metrics_lr["model"] = "Logistic Regression"
    results.append(metrics_lr)

    # Random Forest
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

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_val)
    y_proba_xgb = xgb.predict_proba(X_val)[:, 1]

    metrics_xgb = _compute_metrics(y_val, y_pred_xgb, y_proba_xgb)
    metrics_xgb["model"] = "XGBoost"
    results.append(metrics_xgb)

    df_results = pd.DataFrame(results)[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]

    print("RQ1 – Table 1 (model performance):")
    print(df_results)

    output_table_path = Path(output_table_path)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_table_path, index=False)
    print(f"Saved RQ1 Table 1 to: {output_table_path}")

    return df_results, xgb, feature_names


# ---------------------------------------------------------------------
# RQ1 – Figure 1: feature importance (XGBoost)
# ---------------------------------------------------------------------


def plot_feature_importance(
    xgb_model: XGBClassifier,
    feature_names: List[str],
    output_fig_path: str = "figures/RQ1_Fig1_feature_importance.pdf",
    top_k: int = 10,
) -> None:
    """
    Plot horizontal bar chart of top_k feature importances from XGBoost.
    """
    importances = xgb_model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]

    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Feature importance (Gain)")
    plt.title("RQ1 – Feature importance (XGBoost)")
    plt.tight_layout()

    output_fig_path = Path(output_fig_path)
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig_path)
    plt.close()
    print(f"Saved RQ1 Figure 1 to: {output_fig_path}")


# ---------------------------------------------------------------------
# RQ1 – Table 2 & Figure 2: time-based validation splits (XGBoost)
# ---------------------------------------------------------------------

def _prepare_time_sorted_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the date column is parsed as datetime, drop rows with missing dates,
    and sort by date ascending.
    """
    df = df.copy()
    if DATE_COL not in df.columns:
        raise ValueError(f"Date column '{DATE_COL}' not found in dataset.")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL], format="%d/%m/%Y", errors="coerce"
    )
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def run_time_based_validation(
    df: pd.DataFrame,
    output_table_path: str = "tables/RQ1_Table2_time_based_performance.csv",
    output_fig_path: str = "figures/RQ1_Fig2_time_based_performance.pdf",
    n_splits: int = 4,
) -> pd.DataFrame:
    """
    Create multiple time-based train/test splits and evaluate XGBoost
    performance on each split to study robustness over time.
    """
    # 1) sort and clean by date
    df_time = _prepare_time_sorted_df(df)

    # 2) target and features (use time-specific selector)
    y_all = df_time[TARGET_COL].astype(int)
    X_all_df = _select_features_for_time(df_time)

    # 3) mean imputation for NaNs
    imputer = SimpleImputer(strategy="mean")
    X_all = imputer.fit_transform(X_all_df)

    dates = df_time[DATE_COL]
    unique_dates = np.sort(dates.unique())
    split_points = np.linspace(0.4, 0.8, n_splits)

    records: List[dict] = []

    # 4) train/test splits over different cutoff dates
    for frac in split_points:
        cutoff_idx = int(len(unique_dates) * frac)
        cutoff_date = unique_dates[cutoff_idx]

        train_mask = dates <= cutoff_date
        test_mask = dates > cutoff_date

        # skip degenerate splits
        if train_mask.sum() < 100 or test_mask.sum() < 100:
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)
        y_proba = xgb.predict_proba(X_test)[:, 1]

        metrics = _compute_metrics(y_test, y_pred, y_proba)
        metrics["cutoff_date"] = cutoff_date
        records.append(metrics)

    # 5) save table
    df_time_perf = pd.DataFrame(records)[
        ["cutoff_date", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]

    print("RQ1 – Table 2 (time-based performance):")
    print(df_time_perf)

    output_table_path = Path(output_table_path)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    df_time_perf.to_csv(output_table_path, index=False)
    print(f"Saved RQ1 Table 2 to: {output_table_path}")

    # 6) plot ROC-AUC over time
    plt.figure(figsize=(8, 5))
    plt.plot(df_time_perf["cutoff_date"], df_time_perf["roc_auc"], marker="o")
    plt.xlabel("Train end date (cutoff)")
    plt.ylabel("ROC-AUC")
    plt.title("RQ1 – Time-based validation performance (XGBoost)")
    plt.grid(True)
    plt.tight_layout()

    output_fig_path = Path(output_fig_path)
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig_path)
    plt.close()
    print(f"Saved RQ1 Figure 2 to: {output_fig_path}")

    return df_time_perf


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main(
    dataco_path: str = "data/merged/merged_with_engineered_features.csv",
) -> None:
    print(f"Loading DataCo from: {dataco_path}")
    df = pd.read_csv(dataco_path, low_memory=False)
    print("DataCo shape:", df.shape)

    # RQ1 – Table 1 + Figure 1 (uses merged + engineered features)
    table1_path = "tables/RQ1_Table1_model_performance.csv"
    df_results, xgb_model, feature_names = run_baseline_models(
        df, output_table_path=table1_path
    )

    plot_feature_importance(
        xgb_model,
        feature_names,
        output_fig_path="figures/RQ1_Fig1_feature_importance.pdf",
        top_k=10,
    )

    # RQ1 – Table 2 + Figure 2 (uses original cleaned DataCo file)
    df_dataco_clean = pd.read_csv(
        "data/DataCo_clean_dates_ddmmyyyy.csv", low_memory=False
    )
    run_time_based_validation(
        df_dataco_clean,
        output_table_path="tables/RQ1_Table2_time_based_performance.csv",
        output_fig_path="figures/RQ1_Fig2_time_based_performance.pdf",
        n_splits=4,
    )


if __name__ == "__main__":
    default_dataco = os.environ.get(
        "DE_DATACO_PATH", "data/merged/merged_with_engineered_features.csv"
    )
    main(dataco_path=default_dataco)
