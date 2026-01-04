"""
Feature engineering on merged DataCo + SCMS dataset.

This module creates high-level engineered features that mirror the concepts
described in the project proposal (lead time, route delay rate, ship mode risk,
customer delay history, product delay rate, shipment weight).

Input:
    data/merged/merged_dataco_scms.csv
        - Output of src/feature_engineering/merge_multisource.py

Output:
    data/merged/merged_with_engineered_features.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "Late_delivery_risk"
DATE_COL_ORDER = "order date (DateOrders)"
DATE_COL_SHIPPING = "shipping date (DateOrders)"


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure order/shipping dates are parsed as datetime."""
    df = df.copy()
    if DATE_COL_ORDER in df.columns:
        df[DATE_COL_ORDER] = pd.to_datetime(
            df[DATE_COL_ORDER], format="%d/%m/%Y", errors="coerce"
        )
    if DATE_COL_SHIPPING in df.columns:
        df[DATE_COL_SHIPPING] = pd.to_datetime(
            df[DATE_COL_SHIPPING], format="%d/%m/%Y", errors="coerce"
        )
    return df


def engineer_features(
    merged_path: str = "data/merged/merged_dataco_scms.csv",
    output_path: str = "data/merged/merged_with_engineered_features.csv",
) -> pd.DataFrame:
    """
    Load merged DataCo+SCMS dataset and create engineered features:

    - lead_time_days:
        Time from order date to shipping date (proxy for lead time).
    - ship_mode_risk:
        Historical late-delivery rate for each Shipping Mode.
    - customer_delay_history:
        Historical late-delivery rate per customer.
    - product_delay_rate:
        Historical late-delivery rate per Product Category Id.
    - route_delay_rate:
        Historical late-delivery rate per (Customer Country, Shipping Mode).
    - shipment_weight:
        SCMS-derived average weight for the country (if available).

    The resulting DataFrame is saved and returned.
    """

    print(f"Loading merged dataset from: {merged_path}")
    df = pd.read_csv(merged_path, low_memory=False)
    print("Merged shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in merged dataset."
        )

    # ------------------------------------------------------------------
    # 1. Basic date parsing and lead time
    # ------------------------------------------------------------------
    df = _parse_dates(df)

    if DATE_COL_ORDER in df.columns and DATE_COL_SHIPPING in df.columns:
        df["lead_time_days"] = (
            df[DATE_COL_SHIPPING] - df[DATE_COL_ORDER]
        ).dt.days
    else:
        # Fallback to existing duration columns if needed
        if "Days for shipping (real)" in df.columns:
            df["lead_time_days"] = df["Days for shipping (real)"]
        else:
            df["lead_time_days"] = np.nan

    # ------------------------------------------------------------------
    # 2. Ship mode risk: late-delivery rate per Shipping Mode
    # ------------------------------------------------------------------
    df["Shipping Mode"] = df["Shipping Mode"].astype(str).str.strip()

    shipmode_grp = (
        df.groupby("Shipping Mode", dropna=False)[TARGET_COL]
        .mean()
        .rename("ship_mode_risk")
        .reset_index()
    )

    df = df.merge(
        shipmode_grp,
        on="Shipping Mode",
        how="left",
    )

    # ------------------------------------------------------------------
    # 3. Customer delay history: late-delivery rate per customer
    # ------------------------------------------------------------------
    if "Order Customer Id" in df.columns:
        cust_col = "Order Customer Id"
    elif "Customer Id" in df.columns:
        cust_col = "Customer Id"
    else:
        cust_col = None

    if cust_col is not None:
        cust_grp = (
            df.groupby(cust_col, dropna=False)[TARGET_COL]
            .mean()
            .rename("customer_delay_history")
            .reset_index()
        )
        df = df.merge(
            cust_grp,
            on=cust_col,
            how="left",
        )
    else:
        df["customer_delay_history"] = np.nan

    # ------------------------------------------------------------------
    # 4. Product delay rate: late-delivery rate per Product Category
    # ------------------------------------------------------------------
    prod_key = None
    if "Product Category Id" in df.columns:
        prod_key = "Product Category Id"
    elif "Product Name" in df.columns:
        prod_key = "Product Name"

    if prod_key is not None:
        prod_grp = (
            df.groupby(prod_key, dropna=False)[TARGET_COL]
            .mean()
            .rename("product_delay_rate")
[O            .reset_index()
        )
        df = df.merge(
            prod_grp,
            on=prod_key,
            how="left",
        )
    else:
        df["product_delay_rate"] = np.nan

    # ------------------------------------------------------------------
    # 5. Route delay rate: (Customer Country, Shipping Mode) combination
    # ------------------------------------------------------------------
    df["Customer Country"] = df["Customer Country"].astype(str).str.strip()

    if "Customer Country" in df.columns and "Shipping Mode" in df.columns:
        route_grp = (
            df.groupby(
                ["Customer Country", "Shipping Mode"],
                dropna=False,
            )[TARGET_COL]
            .mean()
            .rename("route_delay_rate")
            .reset_index()
        )
        df = df.merge(
            route_grp,
            on=["Customer Country", "Shipping Mode"],
            how="left",
        )
    else:
        df["route_delay_rate"] = np.nan

    # ------------------------------------------------------------------
    # 6. Shipment weight: SCMS-derived avg_weight renamed
    # ------------------------------------------------------------------
    if "avg_weight" in df.columns:
        df["shipment_weight"] = df["avg_weight"]
    elif "Weight (Kilograms)" in df.columns:
        df["shipment_weight"] = df["Weight (Kilograms)"]
    else:
        df["shipment_weight"] = np.nan

    # ------------------------------------------------------------------
    # Save and return
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved merged dataset with engineered features to: {output_path}")
    print(
        "Created feature columns:",
        [
            "lead_time_days",
            "ship_mode_risk",
            "customer_delay_history",
            "product_delay_rate",
            "route_delay_rate",
            "shipment_weight",
        ],
    )

    return df


if __name__ == "__main__":
    default_merged = os.environ.get(
        "DE_MERGED_INPUT", "data/merged/merged_dataco_scms.csv"
    )
    default_output = os.environ.get(
        "DE_MERGED_FEATURES_OUTPUT",
        "data/merged/merged_with_engineered_features.csv",
    )

    engineer_features(
        merged_path=default_merged,
        output_path=default_output,
    )


