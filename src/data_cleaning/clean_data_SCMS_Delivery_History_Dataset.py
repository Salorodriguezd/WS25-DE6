import pandas as pd  # data frames
import numpy as np


def clean_dataSCMS(df: pd.DataFrame) -> pd.DataFrame:
    df_mod = df.copy()  # Create a copy of the data frame

    # Converting dates into datetime format
    dt = [
        "PQ First Sent to Client Date",
        "PO Sent to Vendor Date",
        "Scheduled Delivery Date",
        "Delivered to Client Date",
        "Delivery Recorded Date",
    ]
    for col in dt:
        df_mod[col] = pd.to_datetime(df_mod[col], errors="coerce")

    # Create Late_delivery_risk target based on delivery dates
    if "Scheduled Delivery Date" in df_mod.columns and "Delivered to Client Date" in df_mod.columns:
        # Drop rows where both critical dates are missing
        df_mod = df_mod.dropna(subset=["Scheduled Delivery Date", "Delivered to Client Date"])

        # Calculate delay in days
        df_mod["delay_days"] = (
            df_mod["Delivered to Client Date"] - df_mod["Scheduled Delivery Date"]
        ).dt.days

        # Create binary Late_delivery_risk target
        # 1 = late (delivered after scheduled date)
        # 0 = on time or early
        df_mod["Late_delivery_risk"] = (df_mod["delay_days"] > 0).astype(int)

    # Delete Dosage column
    if "Dosage" in df_mod.columns:
        df_mod = df_mod.drop(columns=["Dosage"])

    # Delete rows without Shipment Mode
    df_mod = df_mod.dropna(subset=["Shipment Mode"])

    # Weight and freight cost as numeric
    df_mod["Weight (Kilograms)"] = pd.to_numeric(
        df_mod["Weight (Kilograms)"],
        errors="coerce",
    )
    df_mod["Freight Cost (USD)"] = pd.to_numeric(
        df_mod["Freight Cost (USD)"],
        errors="coerce",
    )

    # Categorical variables cleaning
    cat_cols = [
        "Shipment Mode",
        "Country",
        "Vendor",
        "Fulfill Via",
        "Managed By",
        "Vendor INCO Term",
        "Product Group",
    ]

    for col in cat_cols:
        if col in df_mod.columns:
            df_mod[col] = (
                df_mod[col]
                .where(df_mod[col].notna(), np.nan)
                .astype(str)
                .str.strip()
                .str.upper()
            )

    return df_mod

