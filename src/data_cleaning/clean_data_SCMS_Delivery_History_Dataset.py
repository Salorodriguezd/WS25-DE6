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

