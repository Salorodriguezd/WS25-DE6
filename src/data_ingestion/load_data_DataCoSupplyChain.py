import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_dataco_raw() -> pd.DataFrame:
    file_path = DATA_DIR / "raw" / "DataCoSupplyChainDataset.csv"
    df = pd.read_csv(file_path, low_memory=False, encoding="latin-1")
    return df

if __name__ == "__main__":
    df = load_dataco_raw()
    print("Loaded DataCo shape:", df.shape)

