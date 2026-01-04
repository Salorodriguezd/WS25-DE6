# WS25-DE6

# Supply Chain Delivery Delay Prediction from Multi-Source Data

This repository contains the technical implementation for Part II of the Data Engineering project. It implements an end‑to‑end pipeline that integrates two independent supply chain datasets to predict shipment delivery delays, following the approved project proposal.

## 1. Project overview

- **Goal**: Predict whether an order will be delivered late using multi‑source supply chain data.
- **Datasets**:
  - DataCo Supply Chain Dataset (order‑level information)
  - SCMS / Shipment Delivery History Dataset (route‑ and shipment‑level information)
- **Approach**:
  - Clean and harmonize both datasets
  - Merge them into a single analytical table
  - Engineer temporal, route, shipment, product, and customer features
  - Train baseline and boosted models
  - Answer RQ1–RQ3 with reproducible figures and tables

## 2. Research questions

- **RQ1**: How can a machine learning pipeline be designed to remain robust when integrating multi‑source supply chain data?
- **RQ2**: To what extent can independent supply chain datasets be merged to improve delay prediction?
- **RQ3**: What are the most significant predictors of delivery delay?

Each RQ is backed by at least one table and one figure, all generated directly from code.

## 3. Datasets and links

Please download the original datasets from Kaggle before running the pipeline:

- **DataCo Supply Chain Dataset**  
  - Link: [DataCo Supply Chain Dataset:](https://www.kaggle.com/datasets/evilspirit05/datasupplychain)
- **SCMS / Shipment Delivery History Dataset**  
  - Link: [Supply Chain Shipment Pricing Data
](https://www.kaggle.com/datasets/divyeshardeshana/supply-chain-shipment-pricing-data?utm_source=chatgpt.com)

After downloading, place the raw CSVs under `data/raw/` (see below), or adjust the paths in the ingestion/cleaning scripts if needed.

## 4. Repository structure

```text
WS25-DE6/
├── dags/
│   └── project_pipeline_dag.py        # Airflow DAG (pipeline definition)
│
├── src/
│   ├── data_ingestion/               # (optional) raw data download / loading scripts
│   ├── data_cleaning/                # cleaning and preprocessing for DataCo / SCMS
│   ├── feature_engineering/
│   │   ├── merge_multisource.py      # merge DataCo + SCMS into merged_dataco_scms.csv
│   │   └── engineer_features.py      # create engineered features on merged dataset
│   ├── modeling/
│   │   ├── train_models_dataco.py    # RQ1: baseline models, robustness, feature importance
│   │   ├── train_models_rq2.py       # RQ2: single vs multi‑source comparisons
│   │   └── train_models_rq3.py       # RQ3: significant predictors (importance + SHAP)
│   └── evaluation/                   # (optional) helper scripts for exporting tables, etc.
│
├── data/
│   ├── raw/                          # (not tracked) original Kaggle CSVs
│   ├── intermediate/                 # (optional) cleaned versions
│   └── merged/                       # merged and feature‑engineered datasets
│       ├── merged_dataco_scms.csv
│       └── merged_with_engineered_features.csv
│
├── figures/                          # auto‑generated PDF figures
│   ├── RQ1_Fig1_feature_importance.pdf
│   ├── RQ1_Fig2_time_based_performance.pdf
│   ├── RQ2_Fig3_shap_multisource.pdf
│   ├── RQ3_Fig4_relative_importance.pdf
│   ├── RQ3_Fig5_shap_summary.pdf
│   └── RQ3_Fig6_delay_vs_leadtime.pdf
│
├── tables/                           # auto‑generated CSV tables
│   ├── RQ1_Table1_model_performance.csv
│   ├── RQ1_Table2_time_based_performance.csv
│   ├── RQ2_Table3_multisource_vs_single.csv
│   ├── RQ2_Table4_feature_comparison.csv
│   ├── RQ2_Table5_engineered_features.csv
│   └── RQ3_Table6_top_predictors.csv
│
├── requirements.txt
└── README.md
```

## 5. How to run the pipeline

### 5.1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure the raw Kaggle CSV files are downloaded (see Dataset links above) and placed under `data/raw/` or the paths expected by your cleaning scripts.

### 5.2. Generate merged and feature‑engineered data

#### 1) Merge DataCo + SCMS into a single table

```bash
python -m src.feature_engineering.merge_multisource
```
This step reads the cleaned DataCo and SCMS datasets, aligns keys and date formats, and writes the merged multi‑source table:

- `data/merged/merged_dataco_scms.csv`

#### 2) Create engineered features on the merged table

```bash
python -m src.feature_engineering.engineer_features
```

This step takes `merged_dataco_scms.csv` and adds high‑level engineered features such as:

- lead_time_days

- ship_mode_risk

- customer_delay_history

- product_delay_rate

- route_delay_rate

- shipment_weight

The result is saved as:

- `data/merged/merged_with_engineered_features.csv`

All RQ1–RQ3 modeling scripts use this feature‑engineered file as their main input.

## 6. Airflow DAG

The DAG in `dags/project_pipeline_dag.py` mirrors the project pipeline with tasks

The DAG in `dags/project_pipeline_dag.py` mirrors the project pipeline with tasks such as:

- `extract_dataco`, `extract_scms`
- `clean_dataco`, `clean_scms`
- `merge_multisource`
- `engineer_features`
- `train_rq1_models`, `train_rq2_models`, `train_rq3_models`
- `generate_figures_and_tables`

The DAG was developed and tested with:

- **Apache Airflow**: `3.1.5`

To run the DAG locally:

1. Ensure Airflow is installed in your environment.
2. Copy `dags/project_pipeline_dag.py` into your Airflow `dags/` directory (or point `AIRFLOW_HOME/dags` to this repository’s `dags/` folder).
3. Initialize Airflow and start the services:

   ```bash
   airflow db init
   airflow webserver
   airflow scheduler
   ```
4. Open the Airflow UI (by default at `http://localhost:8080`), enable the project DAG, and trigger a run.

The DAG will execute the same steps as the manual CLI commands described in Section 5, and will generate all figures and tables under `figures/` and `tables/`.



## 7. Roles and contributors

- Group: **WS25-DE06**

- **Technical Lead: Seoyeon Kim** – pipeline design, modeling code, figures, tables

- **Documentation & Presentation Lead: Salome Rodriguez Donoso** – report writing, slides, alignment between code and narrative

