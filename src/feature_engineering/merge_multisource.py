import os
from pathlib import Path

import pandas as pd


def build_merged_dataset(
    dataco_path: str = "data/DataCo_clean_dates_ddmmyyyy.csv",
    scms_path: str = "data/SCMS_cleaned.csv",
    output_path: str = "data/merged/merged_dataco_scms.csv",
) -> pd.DataFrame:
    """
    Build a merged dataset where each DataCo order row is augmented with
    statistics computed from the SCMS dataset (aggregated by Country).

    Returns the merged DataFrame and also writes it to `output_path`.
    """

    # --------------------------------------------------------------
    # 1. Load cleaned datasets
    # --------------------------------------------------------------
    print(f"Loading DataCo from: {dataco_path}")
    df_dataco = pd.read_csv(dataco_path, low_memory=False)
    print("DataCo shape:", df_dataco.shape)

    print(f"Loading SCMS from: {scms_path}")
    df_scms = pd.read_csv(
        scms_path,
        parse_dates=[
            "Scheduled Delivery Date",
            "Delivered to Client Date",
        ],
    )
    print("SCMS shape:", df_scms.shape)

    # --------------------------------------------------------------
    # 2. Build SCMS features (aggregated by Country)
    # --------------------------------------------------------------
    df_scms["transit_time_days"] = (
        df_scms["Delivered to Client Date"]
        - df_scms["Scheduled Delivery Date"]
    ).dt.days

    df_scms["is_delayed"] = (df_scms["transit_time_days"] > 0).astype(int)

    df_scms["Country"] = df_scms["Country"].astype(str).str.strip()

    scms_stats = (
        df_scms
        .groupby("Country", dropna=False)
        .agg(
            avg_transit_time_days=("transit_time_days", "mean"),
            delay_rate=("is_delayed", "mean"),
            avg_freight_cost=("Freight Cost (USD)", "mean"),
            avg_weight=("Weight (Kilograms)", "mean"),
            n_shipments=("Country", "count"),
        )
        .reset_index()
    )

    print("SCMS stats shape:", scms_stats.shape)

    # --------------------------------------------------------------
    # 3. Prepare DataCo and merge
    # --------------------------------------------------------------
    df_dataco["Customer Country"] = (
        df_dataco["Customer Country"].astype(str).str.strip()
    )

    df_merged = df_dataco.merge(
        scms_stats,
        left_on="Customer Country",
        right_on="Country",
        how="left",
        suffixes=("", "_scms"),
    )

    print("Merged shape:", df_merged.shape)

    # Check the first 5 rows
    cols_check = [
        "Customer Country",
        "Shipping Mode",
        "avg_transit_time_days",
        "delay_rate",
        "avg_freight_cost",
        "avg_weight",
        "n_shipments",
    ]
    print("Merged sample (first 5 rows):")
    print(df_merged[cols_check].head().to_markdown(index=False))

    # --------------------------------------------------------------
    # 4. Save to CSV
    # --------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged dataset to: {output_path}")

    return df_merged

# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Run from repo root with:
    #   python -m src.feature_engineering.merge_dataco_scms
    default_dataco = os.environ.get(
        "DE_DATACO_PATH", "data/DataCo_clean_dates_ddmmyyyy.csv"
    )
    default_scms = os.environ.get(
        "DE_SCMS_PATH", "data/SCMS_cleaned.csv"
    )
    default_output = os.environ.get(
        "DE_MERGED_OUTPUT", "data/merged/merged_dataco_scms.csv"
    )

    build_merged_dataset(
        dataco_path=default_dataco,
        scms_path=default_scms,
        output_path=default_output,
    )

