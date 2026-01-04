from data_ingestion.load_data import load_dataSCMS_raw
from data_cleaning.clean_dataco import clean_dataco
from pathlib import Path

DATA_DIR = Path("data2")

if __name__ == "__main__":
    # 1) ingest
    df_raw2 = load_dataSCMS_raw()

    # 2) clean
    df_clean2 = clean_dataSCMS(df_raw2)

    # 3) save intermediate outputs
    output_path = DATA_DIR / "intermediate" / "DataSCMS_clean.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print("Saved cleaned file to:", output_path)
