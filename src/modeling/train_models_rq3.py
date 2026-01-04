"""
Modeling for RQ3: significant predictors of delivery delay.

Outputs (saved under ./tables and ./figures):

- tables/RQ3_Table6_top_predictors.csv
    Top predictors ranked by importance score from a tree-based model.

- figures/RQ3_Fig4_relative_importance.pdf
    Bar chart of relative feature importance (temporal, route-level, shipment, customer, etc.).

- figures/RQ3_Fig5_shap_summary.pdf
    SHAP summary plot showing direction and magnitude of predictor influence.

- figures/RQ3_Fig6_delay_vs_leadtime.pdf
    Delay probability by lead time bucket.
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
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.modeling.train_models_dataco import _select_features as _select_features_rq1

TARGET_COL = "Late_delivery_risk"


# ---------------------------------------------------------------------
# Utility: metrics and training
# ---------------------------------------------------------------------


def _compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _train_xgb_for_rq3(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train an XGBoost classifier on the merged dataset using the same
    feature selection logic as RQ1.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = df[TARGET_COL].astype(int)
    X_df = _select_features_rq1(df)
    feature_names = X_df.columns.tolist()
    X = X_df.values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y.values, test_size=test_size, random_state=random_state, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

[O    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    metrics = _compute_metrics(y_val, y_pred, y_proba)

    return model, X_train, X_val, y_train, y_val, feature_names, metrics


# ---------------------------------------------------------------------
# RQ3 – Table 6: top predictors ranked by importance
# ---------------------------------------------------------------------


def build_rq3_table6(
    df: pd.DataFrame,
    output_csv: str = "tables/RQ3_Table6_top_predictors.csv",
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Train XGBoost on the merged dataset and export the top-k predictors
    ranked by feature importance.
    """
    model, X_train, X_val, y_train, y_val, feature_names, metrics = _train_xgb_for_rq3(
        df
    )

    importances = model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]

    top_idx = idx_sorted[:top_k]
    rows = []
    for i in top_idx:
        rows.append(
            {
                "predictor": feature_names[i],
                "importance_score": float(importances[i]),
            }
        )

    df_top = pd.DataFrame(rows)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(out_path, index=False)
    print(f"Saved RQ3 Table 6 (top predictors) to: {out_path}")

    return df_top


# ---------------------------------------------------------------------
# RQ3 – Figure 4: relative importance of predictors
# ---------------------------------------------------------------------


def plot_rq3_feature_importance(
    df_top: pd.DataFrame,
    output_fig: str = "figures/RQ3_Fig4_relative_importance.pdf",
) -> None:
    """
    Plot a horizontal bar chart for the top predictors from Table 6.
    """
    predictors = df_top["predictor"].tolist()
    importances = df_top["importance_score"].values

    # sort for nicer plotting (small to large, bottom to top)
    order = np.argsort(importances)
    predictors_sorted = [predictors[i] for i in order]
    importances_sorted = importances[order]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(predictors_sorted)), importances_sorted)
[I    plt.yticks(range(len(predictors_sorted)), predictors_sorted)
    plt.xlabel("Importance score")
    plt.title("RQ3 – Relative importance of predictors (XGBoost)")
    plt.tight_layout()

    out_path = Path(output_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RQ3 Figure 4 (relative importance) to: {out_path}")


# ---------------------------------------------------------------------
# RQ3 – Figure 5: SHAP summary plot
# ---------------------------------------------------------------------


def plot_rq3_shap_summary(
    df: pd.DataFrame,
    output_fig: str = "figures/RQ3_Fig5_shap_summary.pdf",
) -> None:
    """
    Train XGBoost on the merged dataset and plot a SHAP summary
    (beeswarm) plot to show direction and magnitude of feature influence.
    """
    model, X_train, X_val, y_train, y_val, feature_names, metrics = _train_xgb_for_rq3(
        df
    )

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
    print(f"Saved RQ3 Figure 5 (SHAP summary) to: {out_path}")


# ---------------------------------------------------------------------
# RQ3 – Figure 6: delay probability by lead time bucket
# ---------------------------------------------------------------------


def plot_rq3_delay_vs_leadtime(
    df: pd.DataFrame,
    lead_time_col: str = "lead_time_days",
    output_fig: str = "figures/RQ3_Fig6_delay_vs_leadtime.pdf",
) -> None:
    """
    Compute delay probability by lead time bucket and plot it.

    Buckets are defined over lead_time_days; if that column is missing,
    this function will raise an error.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    if lead_time_col not in df.columns:
        raise ValueError(
            f"Lead time column '{lead_time_col}' not found in dataset. "
            "Ensure engineer_features.py created it."
        )

    tmp = df[[lead_time_col, TARGET_COL]].copy()
    tmp = tmp.dropna(subset=[lead_time_col])

    # define buckets (you can adjust thresholds if needed)
    bins = [-np.inf, 3, 5, 7, 10, np.inf]
    labels = ["<=3", "3-5", "5-7", "7-10", ">10"]
    tmp["lead_time_bucket"] = pd.cut(tmp[lead_time_col], bins=bins, labels=labels)

    bucket_stats = (
        tmp.groupby("lead_time_bucket", observed=False)[TARGET_COL]
           .mean()
           .rename("delay_probability")
           .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        bucket_stats["lead_time_bucket"].astype(str),
        bucket_stats["delay_probability"],
        marker="o",
    )
    plt.xlabel("Lead time bucket (days)")
    plt.ylabel("Delay probability")
    plt.title("RQ3 – Delay probability by lead time bucket")
    plt.grid(True)
    plt.tight_layout()

    out_path = Path(output_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RQ3 Figure 6 (delay vs lead time) to: {out_path}")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main(
    merged_path: str = "data/merged/merged_with_engineered_features.csv",
) -> None:
    print(f"Loading merged dataset for RQ3 from: {merged_path}")
    df = pd.read_csv(merged_path, low_memory=False)
    print("Merged dataset shape:", df.shape)

    # Table 6
    df_top = build_rq3_table6(df)

    # Figure 4: relative importance bar chart
    plot_rq3_feature_importance(df_top)

    # Figure 5: SHAP summary plot
    plot_rq3_shap_summary(df)

    # Figure 6: delay probability by lead time bucket
    plot_rq3_delay_vs_leadtime(df)


if __name__ == "__main__":
    main()


