from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# --- make WS25-DE6 project importable ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- import project code ---
from src.data_ingestion.load_data_DataCoSupplyChain import load_dataco_raw
from src.data_ingestion.load_data_DataSCMS import load_dataSCMS_raw
from src.data_cleaning.clean_data_DataCoSupplyChain import clean_dataco
from src.data_cleaning.clean_data_SCMS_Delivery_History_Dataset import clean_dataSCMS
from src.feature_engineering.merge_multisource import build_merged_dataset
from src.feature_engineering.engineer_features import engineer_features
from src.modeling.train_models_dataco import run_baseline_models
from src.modeling.train_models_rq2 import run_rq2_table3
from src.modeling.train_models_rq3 import main as run_rq3_all
from src.evaluation.export_tables_to_excel import main as export_tables_to_excel

import pandas as pd


default_args = {
    "owner": "de_project_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="project_pipeline_dag",
    description="End-to-end pipeline for supply chain delivery delay project",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,  # manual trigger
    catchup=False,
    tags=["data_engineering", "course_project"],
)


# -----------------------------
# 1. Ingestion + Cleaning
# -----------------------------
def ingest_and_clean_dataco():
    data_dir = Path("data")
    df_raw = load_dataco_raw()
    df_clean = clean_dataco(df_raw)
    out_path = data_dir / "intermediate" / "DataCo_clean_dates_ddmmyyyy.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"Saved cleaned DataCo to: {out_path}")


def ingest_and_clean_scms():
    data_dir = Path("data")
    df_raw = load_dataSCMS_raw()
    df_clean = clean_dataSCMS(df_raw)
    out_path = data_dir / "intermediate" / "SCMS_cleaned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"Saved cleaned SCMS to: {out_path}")


ingest_dataco = PythonOperator(
    task_id="ingest_and_clean_dataco",
    python_callable=ingest_and_clean_dataco,
    dag=dag,
)

ingest_scms = PythonOperator(
    task_id="ingest_and_clean_scms",
    python_callable=ingest_and_clean_scms,
    dag=dag,
)

# -----------------------------
# 2. Merge + Feature engineering
# -----------------------------
def merge_dataco_scms():
    dataco_path = "data/intermediate/DataCo_clean_dates_ddmmyyyy.csv"
    scms_path = "data/intermediate/SCMS_cleaned.csv"
    output_path = "data/merged/merged_dataco_scms.csv"
    build_merged_dataset(
        dataco_path=dataco_path,
        scms_path=scms_path,
        output_path=output_path,
    )


merge_sources = PythonOperator(
    task_id="merge_dataco_scms",
    python_callable=merge_dataco_scms,
    dag=dag,
)


def engineer_features_task():
    engineer_features(
        merged_path="data/merged/merged_dataco_scms.csv",
        output_path="data/merged/merged_with_engineered_features.csv",
    )


feature_engineering = PythonOperator(
    task_id="engineer_features",
    python_callable=engineer_features_task,
    dag=dag,
)

# -----------------------------
# 3. Modeling for RQ1, RQ2, RQ3
# -----------------------------
def run_rq1_models():
    df_dataco = pd.read_csv(
        "data/intermediate/DataCo_clean_dates_ddmmyyyy.csv", low_memory=False
    )
[O    run_baseline_models(df_dataco)


rq1_modeling = PythonOperator(
    task_id="run_rq1_models",
    python_callable=run_rq1_models,
    dag=dag,
)


def run_rq2_models():
    run_rq2_table3(
        dataco_path="data/intermediate/DataCo_clean_dates_ddmmyyyy.csv",
        scms_path="data/intermediate/SCMS_cleaned.csv",
        merged_path="data/merged/merged_with_engineered_features.csv",
        output_csv="tables/RQ2_Table3_multisource_vs_single.csv",
    )


rq2_modeling = PythonOperator(
    task_id="run_rq2_models",
    python_callable=run_rq2_models,
    dag=dag,
)


def run_rq3_models():
    run_rq3_all(merged_path="data/merged/merged_with_engineered_features.csv")


rq3_modeling = PythonOperator(
    task_id="run_rq3_models",
    python_callable=run_rq3_models,
    dag=dag,
)

# -----------------------------
# 4. Export tables to Excel
# -----------------------------
def export_tables():
    export_tables_to_excel(tables_dir="tables")


export_excel = PythonOperator(
    task_id="export_tables_to_excel",
    python_callable=export_tables,
    dag=dag,
)

# -----------------------------
# Dependencies
# -----------------------------
[ingest_dataco, ingest_scms] >> merge_sources >> feature_engineering
feature_engineering >> [rq1_modeling, rq2_modeling, rq3_modeling]
[rq1_modeling, rq2_modeling, rq3_modeling] >> export_excel


