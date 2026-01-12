from src.data_ingestion.load_data_DataSCMS import load_dataSCMS_raw
from src.data_cleaning.clean_data_SCMS_Delivery_History_Dataset import clean_dataSCMS
from pathlib import Path

DATA_DIR = Path("data")

if __name__ == "__main__":
    # 1) ingest
    df_raw = load_dataSCMS_raw()

    # 2) clean and create Late_delivery_risk
    df_clean = clean_dataSCMS(df_raw)

    # 3) save cleaned output
    output_path = DATA_DIR / "SCMS_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print("Saved cleaned file to:", output_path)

