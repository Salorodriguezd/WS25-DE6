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
    SHAP summary plot showing predictive power of multi-source features.
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

    # Drop obvious IDs and string descriptors if present
    drop_cols = [
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
[O        "Delivery Recorded Date",
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
[I    output_csv: str = "tables/RQ2_Table3_multisource_vs_single.csv",
) -> pd.DataFrame:
    """
    Train the same classifier (XGBoost) on:
    - DataCo only
    - Merged multi-source dataset

    and include SCMS as a row with N/A metrics (no delay label available).

    The goal is to match the proposal structure:
    DataCo vs SCMS vs DataCo+SCMS. [file:116]
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
    # 2) SCMS only (no label available → metrics set to N/A)
    # ------------------------------------------------------------------
    df_scms = pd.read_csv(scms_path, low_memory=False)
    _ = _select_features_scms(df_scms)  # used only to illustrate feature scope

    records.append(
        {
            "dataset": "SCMS",
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }
    )

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

    This table follows the structure in the proposal (Table 4). [file:116]
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

    This table mirrors the proposal's Table 5. [file:116]
    """
    rows = [
        {
            "Feature Name": "transit_time_days",
            "Original Columns Used": "actual_delivery_date, planned_delivery_date",
            "Source Dataset": "SCMS",
            "Feature Type": "Temporal",
            "Description": "Days difference between planned and actual delivery",
            "Predictive Purpose": "Detect systematic transit delays",
        },
        {
            "Feature Name": "route_delay_rate_90d",
            "Original Columns Used": "origin, destination, actual_delivery_date, planned_delivery_date",
            "Source Dataset": "SCMS",
            "Feature Type": "Historical",
            "Description": "Share of delays on the same route in the last 90 days",
            "Predictive Purpose": "Capture structural geographic risk",
        },
        {
            "Feature Name": "avg_route_cost",
            "Original Columns Used": "shipment_cost, origin, destination",
            "Source Dataset": "SCMS",
            "Feature Type": "Statistical",
            "Description": "Average shipping cost per route",
            "Predictive Purpose": "Proxy for logistics constraints and complexity",
        },
        {
            "Feature Name": "transit_time_std",
            "Original Columns Used": "actual_delivery_date, origin, destination",
            "Source Dataset": "SCMS",
            "Feature Type": "Variability",
            "Description": "Variance of historic transit times",
            "Predictive Purpose": "Identify unstable or unpredictable routes",
        },
        {
            "Feature Name": "order_processing_time",
            "Original Columns Used": "order_date, shipping_date",
            "Source Dataset": "DataCo",
            "Feature Type": "Temporal",
            "Description": "Time from order creation to shipping",
            "Predictive Purpose": "Measure internal operational lead time",
        },
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
# RQ2 – Figure 3: SHAP summary for multi-source model
# ---------------------------------------------------------------------


def plot_rq2_shap_multisource(
    merged_path: str = "data/merged/merged_with_engineered_features.csv",
    output_fig: str = "figures/RQ2_Fig3_shap_multisource.pdf",
) -> None:
    """
    Train XGBoost on merged multi-source data and create a SHAP summary plot
    for feature importance and directional effects (Figure 3). [file:116]
    """
    df = pd.read_csv(merged_path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in merged dataset."
        )

    y = df[TARGET_COL].astype(int)
    X_df = _select_features_merged(df)
    feature_names = X_df.columns.tolist()
    X = X_df.values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y.values, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
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
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_val, feature_names=feature_names, show=False
    )

    out_path = Path(output_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RQ2 Figure 3 (SHAP summary) to: {out_path}")


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

    # Figure 3: SHAP on multi-source model
    plot_rq2_shap_multisource()


if __name__ == "__main__":
    main()

