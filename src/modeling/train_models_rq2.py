"""
Modeling for RQ2: impact of multi-source integration.

Outputs (saved under ./tables and ./figures):

- tables/RQ2_Table3_multisource_vs_single.csv
    Comparative performance of using single-source vs multi-source data.

- tables/RQ2_Table4_feature_comparison.csv
    Feature-type availability before vs after dataset integration.

- tables/RQ2_Table5_engineered_features.csv
    Catalog of engineered features resulting from merging multi-source tables.

- figures/RQ2_Fig3_shap_multisource.pdf
    SHAP comparison: Baseline (DataCo only) vs Integrated (multi-source) models.
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET_COL = "Late_delivery_risk"


# ---------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------


def _select_features_dataco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature selection for the DataCo-only dataset.

    - Drop target and obvious ID/postal-code columns.
    - Keep only numeric features.
    - Drop columns that are entirely NaN.
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
    X_num = X.select_dtypes(include=[np.number])
    X_num = X_num.loc[:, ~X_num.isna().all()]

    return X_num


def _select_features_scms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature selection for the SCMS_cleaned dataset.

    Since there is no delay label in SCMS_cleaned, this selector is only used
    for descriptive comparison (not for supervised training).
    """
    df = df.copy()

    # Drop target and intermediate columns used to create target
    drop_cols = [
        TARGET_COL,
        "delay_days",
        "ID",
        "Project Code",
        "PQ #",
        "PO / SO #",
        "ASN/DN #",
        "Country",
        "Managed By",
        "Fulfill Via",
        "Vendor INCO Term",
        "Shipment Mode",
        "PQ First Sent to Client Date",
        "PO Sent to Vendor Date",
        "Scheduled Delivery Date",
        "Delivered to Client Date",
        "Delivery Recorded Date",
        "Product Group",
        "Sub Classification",
        "Vendor",
        "Item Description",
        "Molecule/Test Type",
        "Brand",
        "Dosage Form",
        "Unit of Measure (Per Pack)",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    X_num = X.select_dtypes(include=[np.number])
    X_num = X_num.loc[:, ~X_num.isna().all()]

    return X_num


def _select_features_merged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature selection for the merged multi-source dataset
    (merged_with_engineered_features.csv).

    Reuses the RQ1 selector so engineered features are kept automatically.
    """
    from src.modeling.train_models_dataco import _select_features as _sel_rq1

    X_num = _sel_rq1(df)
    return X_num


# ---------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------


def _compute_cls_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------
# RQ2 – Table 3: single-source vs multi-source performance
# ---------------------------------------------------------------------


def run_rq2_table3(
    dataco_path: str = "data/DataCo_clean_dates_ddmmyyyy.csv",
    scms_path: str = "data/SCMS_cleaned.csv",
    merged_path: str = "data/merged/merged_with_engineered_features.csv",
    output_csv: str = "tables/RQ2_Table3_multisource_vs_single.csv",
) -> pd.DataFrame:
    """
    Train the same classifier (XGBoost) on:
    - DataCo only
    - Merged multi-source dataset

    and include SCMS as a row with N/A metrics (no delay label available).

    The goal is to match the proposal structure:
    DataCo vs SCMS vs DataCo+SCMS.
    """
    records: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # 1) DataCo only
    # ------------------------------------------------------------------
    df_dataco = pd.read_csv(dataco_path, low_memory=False)
    if TARGET_COL not in df_dataco.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataCo dataset.")

    y = df_dataco[TARGET_COL].astype(int)
    X_df = _select_features_dataco(df_dataco)
    X = X_df.values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y.values, test_size=0.2, random_state=42, stratify=y
    )

    xgb_dataco = XGBClassifier(
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
    xgb_dataco.fit(X_train, y_train)

    y_pred = xgb_dataco.predict(X_val)
    metrics = _compute_cls_metrics(y_val, y_pred)
    metrics.update({"dataset": "DataCo"})
    records.append(metrics)

    # ------------------------------------------------------------------
    # 2) SCMS only
    # ------------------------------------------------------------------
    df_scms = pd.read_csv(scms_path, low_memory=False)
    if TARGET_COL not in df_scms.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in SCMS_cleaned dataset."
        )

    y_s = df_scms[TARGET_COL].astype(int)
    X_s_df = _select_features_scms(df_scms)
    X_s = X_s_df.values

    X_train, X_val, y_train, y_val = train_test_split(
        X_s, y_s.values, test_size=0.2, random_state=42, stratify=y_s
    )

    xgb_scms = XGBClassifier(
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
    xgb_scms.fit(X_train, y_train)

    y_pred = xgb_scms.predict(X_val)
    metrics = _compute_cls_metrics(y_val, y_pred)
    metrics.update({"dataset": "SCMS"})
    records.append(metrics)

    # ------------------------------------------------------------------
    # 3) Merged multi-source dataset
    # ------------------------------------------------------------------
    df_merged = pd.read_csv(merged_path, low_memory=False)
    if TARGET_COL not in df_merged.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in merged dataset."
        )

    y_m = df_merged[TARGET_COL].astype(int)
    X_m_df = _select_features_merged(df_merged)
    X_m = X_m_df.values

    X_train, X_val, y_train, y_val = train_test_split(
        X_m, y_m.values, test_size=0.2, random_state=42, stratify=y_m
    )

    xgb_merged = XGBClassifier(
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
    xgb_merged.fit(X_train, y_train)

    y_pred = xgb_merged.predict(X_val)
    metrics = _compute_cls_metrics(y_val, y_pred)
    metrics.update({"dataset": "DataCo+SCMS"})
    records.append(metrics)

    # ------------------------------------------------------------------
    # Save Table 3
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(records)[
        ["dataset", "accuracy", "precision", "recall", "f1"]
    ]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"Saved RQ2 Table 3 to: {out_path}")

    return df_results


# ---------------------------------------------------------------------
# RQ2 – Table 4: feature comparison before/after integration
# ---------------------------------------------------------------------


def build_rq2_table4(
    output_csv: str = "tables/RQ2_Table4_feature_comparison.csv",
) -> pd.DataFrame:
    """
    Schema-level comparison: which feature types are available in each dataset.

    This table follows the structure in the proposal (Table 4).
    """
    rows = [
        {
            "Feature Type": "Customer behavior history",
            "DataCo": "Yes",
            "SCMS": "No",
            "Merged Data": "Yes",
        },
        {
            "Feature Type": "Route transit variability",
            "DataCo": "No",
            "SCMS": "Yes",
            "Merged Data": "Yes",
        },
        {
            "Feature Type": "Seasonal demand",
            "DataCo": "Yes",
            "SCMS": "No",
            "Merged Data": "Yes",
        },
        {
            "Feature Type": "Cost vs delay correlation",
            "DataCo": "No",
            "SCMS": "Yes",
            "Merged Data": "Yes",
        },
        {
            "Feature Type": "Product-category risk",
            "DataCo": "Yes",
            "SCMS": "No",
            "Merged Data": "Yes",
        },
    ]
    df = pd.DataFrame(rows)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved RQ2 Table 4 to: {out_path}")
    return df


# ---------------------------------------------------------------------
# RQ2 – Table 5: engineered features catalog
# ---------------------------------------------------------------------


def build_rq2_table5(
    output_csv: str = "tables/RQ2_Table5_engineered_features.csv",
) -> pd.DataFrame:
    """
    Engineered features resulting from merging multi-source tables.

    This table mirrors the proposal's Table 5.
    """
    rows = [
                {
            "Feature Name": "customer_delay_history",
            "Original Columns Used": "customer_id, Late_delivery_risk",
            "Source Dataset": "DataCo",
            "Feature Type": "Historical",
            "Description": "Share of late deliveries per customer",
            "Predictive Purpose": "Capture recurring customer-specific issues",
        },
        {
            "Feature Name": "product_delay_rate",
            "Original Columns Used": "product_category, Late_delivery_risk",
            "Source Dataset": "DataCo",
            "Feature Type": "Product-level",
            "Description": "Delay rate grouped by product type or category",
            "Predictive Purpose": "Identify product categories that are harder to ship",
        },
        {
            "Feature Name": "ship_mode_risk",
            "Original Columns Used": "shipping_mode, Late_delivery_risk",
            "Source Dataset": "DataCo",
            "Feature Type": "Categorical",
            "Description": "Late-delivery risk associated with transportation mode",
            "Predictive Purpose": "Quantify impact of transport mode choice",
        },
        {
            "Feature Name": "seasonality_week",
            "Original Columns Used": "order_date",
            "Source Dataset": "Both",
            "Feature Type": "Temporal",
            "Description": "Week-of-year derived from order date",
            "Predictive Purpose": "Capture peak seasons and demand cycles",
        },
    ]
    df = pd.DataFrame(rows)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved RQ2 Table 5 to: {out_path}")
    return df


# ---------------------------------------------------------------------
# RQ2 – Figure 3: SHAP comparison (Baseline vs Integrated)
# ---------------------------------------------------------------------


def plot_rq2_shap_multisource(
    dataco_path: str = "data/DataCo_clean_dates_ddmmyyyy.csv",
    merged_path: str = "data/merged/merged_with_engineered_features.csv",
    output_fig: str = "figures/RQ2_Fig3_shap_multisource.pdf",
) -> None:
    """
    Train XGBoost on:
    1) DataCo only (baseline)
    2) Merged multi-source data (integrated)
    
    Create side-by-side SHAP bar plots to compare feature importance
    and show the impact of multi-source integration.
    """
    # ------------------------------------------------------------------
    # 1) Baseline: DataCo only
    # ------------------------------------------------------------------
    print("Training baseline model (DataCo only)...")
    df_dataco = pd.read_csv(dataco_path, low_memory=False)
    if TARGET_COL not in df_dataco.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataCo dataset.")
    
    y_base = df_dataco[TARGET_COL].astype(int)
    X_base_df = _select_features_dataco(df_dataco)
    feature_names_base = X_base_df.columns.tolist()
    X_base = X_base_df.values
    
    X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
        X_base, y_base.values, test_size=0.2, random_state=42, stratify=y_base
    )
    
    model_base = XGBClassifier(
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
    model_base.fit(X_train_base, y_train_base)
    
    explainer_base = shap.TreeExplainer(model_base)
    shap_values_base = explainer_base(X_val_base)
    
    # ------------------------------------------------------------------
    # 2) Integrated: merged multi-source data
    # ------------------------------------------------------------------
    print("Training integrated model (DataCo + SCMS + Engineered)...")
    df_merged = pd.read_csv(merged_path, low_memory=False)
    if TARGET_COL not in df_merged.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in merged dataset.")
    
    y_merged = df_merged[TARGET_COL].astype(int)
    X_merged_df = _select_features_merged(df_merged)
    feature_names_merged = X_merged_df.columns.tolist()
    X_merged = X_merged_df.values
    
    X_train_merged, X_val_merged, y_train_merged, y_val_merged = train_test_split(
        X_merged, y_merged.values, test_size=0.2, random_state=42, stratify=y_merged
    )
    
    model_merged = XGBClassifier(
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
    model_merged.fit(X_train_merged, y_train_merged)
    
    explainer_merged = shap.TreeExplainer(model_merged)
    shap_values_merged = explainer_merged(X_val_merged)
    
    # ------------------------------------------------------------------
    # 3) Side-by-side comparison plot
    # ------------------------------------------------------------------
    print("Creating SHAP comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline (left)
    plt.subplot(1, 2, 1)
    shap.summary_plot(
        shap_values_base,
        X_val_base,
        feature_names=feature_names_base,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    ax1 = plt.gca()
    ax1.set_title("Baseline (DataCo only)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("mean(|SHAP value|)", fontsize=12)
    
    # Integrated (right)
    plt.subplot(1, 2, 2)
    shap.summary_plot(
        shap_values_merged,
        X_val_merged,
        feature_names=feature_names_merged,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    ax2 = plt.gca()
    ax2.set_title("Integrated (DataCo + SCMS + Engineered)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("mean(|SHAP value|)", fontsize=12)
    
    plt.tight_layout()
    
    out_path = Path(output_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RQ2 Figure 3 (SHAP comparison: Baseline vs Integrated) to: {out_path}")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main() -> None:
    # Table 3: single vs multi-source performance
    run_rq2_table3()

    # Table 4: feature comparison
    build_rq2_table4()

    # Table 5: engineered features
    build_rq2_table5()

    # Figure 3: SHAP comparison (Baseline vs Integrated)
    plot_rq2_shap_multisource(
        dataco_path="data/DataCo_clean_dates_ddmmyyyy.csv",
        merged_path="data/merged/merged_with_engineered_features.csv"
    )


if __name__ == "__main__":
    main()

