from src.data_ingestion.load_data_DataCoSupplyChain import load_dataco_raw
from src.data_cleaning.clean_data_DataCoSupplyChain import clean_dataco
from pathlib import Path

DATA_DIR = Path("data")

if __name__ == "__main__":
    # 1) ingest
    df_raw = load_dataco_raw()

    # 2) clean
    df_clean = clean_dataco(df_raw)

    # 3) save cleaned output
    output_path = DATA_DIR / "DataCo_clean_dates_ddmmyyyy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print("Saved cleaned file to:", output_path)

