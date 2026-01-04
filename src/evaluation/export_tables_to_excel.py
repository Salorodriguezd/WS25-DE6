
from pathlib import Path
import pandas as pd

TABLES = [
    "RQ1_Table1_model_performance.csv",
    "RQ1_Table2_time_based_performance.csv",
    "RQ2_Table3_multisource_vs_single.csv",
    "RQ2_Table4_feature_comparison.csv",
    "RQ2_Table5_engineered_features.csv",
    "RQ3_Table6_top_predictors.csv",
]

def main(tables_dir: str = "tables") -> None:
    base = Path(tables_dir)
    for name in TABLES:
        csv_path = base / name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        xlsx_path = base / name.replace(".csv", ".xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"Saved {xlsx_path}")

if __name__ == "__main__":
    main()
    
