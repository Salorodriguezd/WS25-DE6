import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_dataSCMS_raw() -> pd.DataFrame:
    file_path = DATA_DIR / "raw" / "SCMS_Delivery_History_Dataset.csv"
    df = pd.read_csv(file_path, low_memory=False, encoding="latin-1")
    return df

if __name__ == "__main__":
    df = load_dataSCMS_raw()
    print("Loaded DataSCMS shape:", df.shape)
