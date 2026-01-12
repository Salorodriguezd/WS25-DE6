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

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
```

Make sure the raw Kaggle CSV files are downloaded (see Dataset links in Section 3) and placed under `data/raw/`:

- `data/raw/DataCoSupplyChainDataset.csv`
- `data/raw/SCMS_Delivery_History_Dataset.csv`

If your raw files have different names, adjust the paths in the ingestion scripts (`src/data_ingestion/load_data_*.py`).

***

### 5.2. Data cleaning

Before merging or modeling, each dataset must be cleaned independently.

#### Clean DataCo dataset

```bash
python -m src.data_ingestion.run_DataCo_pipeline
```

This step:

- Loads `data/raw/DataCoSupplyChainDataset.csv`
- Parses dates, removes duplicates, handles missing values
- Saves the cleaned dataset to `data/DataCo_clean_dates_ddmmyyyy.csv`


#### Clean SCMS dataset

```bash
python -m src.data_ingestion.run_DataSCMS_pipeline
```

This step:

- Loads `data/raw/SCMS_Delivery_History_Dataset.csv`
- Parses delivery dates and creates the `Late_delivery_risk` target column based on:
    - `Late_delivery_risk = 1` if `Delivered to Client Date > Scheduled Delivery Date`
    - `Late_delivery_risk = 0` otherwise (on time or early)
- Cleans categorical variables and numeric columns
- Saves the cleaned dataset to `data/SCMS_cleaned.csv`

After this step, both cleaned datasets are ready for integration.

***

### 5.3. Feature engineering

Now that both datasets are cleaned, we can merge them and create engineered features.

#### 1) Merge DataCo + SCMS into a single table

```bash
python -m src.feature_engineering.merge_multisource
```

This step:

- Reads `data/DataCo_clean_dates_ddmmyyyy.csv` and `data/SCMS_cleaned.csv`
- Performs a left join on `Country` (or route-level aggregates)
- Aligns date formats and keys
- Saves the merged multi-source table to:
    - `data/merged/merged_dataco_scms.csv`


#### 2) Create engineered features on the merged table

```bash
python -m src.feature_engineering.engineer_features
```

This step:

- Loads `data/merged/merged_dataco_scms.csv`
- Creates high-level engineered features:
    - `lead_time_days` – time from order to shipping
    - `ship_mode_risk` – historical late-delivery rate per shipping mode
    - `customer_delay_history` – late-delivery rate per customer
    - `product_delay_rate` – delay rate per product category
    - `route_delay_rate` – delay rate per (country, shipping mode) combination
    - `shipment_weight` – SCMS-derived average weight
- Saves the feature-engineered dataset to:
    - `data/merged/merged_with_engineered_features.csv`

All RQ1–RQ3 modeling scripts use this file as their main input.

***

### 5.4. Modeling (RQ1, RQ2, RQ3)

With the feature-engineered dataset ready, run each modeling script to generate figures and tables.

#### RQ1 – Robust pipeline and baseline models

```bash
python -m src.modeling.train_models_dataco
```

**Outputs**:

- `tables/RQ1_Table1_model_performance.csv` – Accuracy/Recall/F1 for Logistic Regression, Random Forest, XGBoost
- `tables/RQ1_Table2_time_based_performance.csv` – ROC-AUC across time-based splits
- `figures/RQ1_Fig1_feature_importance.pdf` – Top 10 features from Random Forest
- `figures/RQ1_Fig2_time_based_performance.pdf` – Line plot of ROC-AUC over time


#### RQ2 – Single-source vs multi-source comparison

```bash
python -m src.modeling.train_models_rq2
```

**Outputs**:

- `tables/RQ2_Table3_multisource_vs_single.csv` – Performance comparison (DataCo vs SCMS vs Merged)
- `tables/RQ2_Table4_feature_comparison.csv` – Feature type availability matrix
- `tables/RQ2_Table5_engineered_features.csv` – Catalog of engineered features
- `figures/RQ2_Fig3_shap_multisource.pdf` – SHAP summary plot


#### RQ3 – Significant predictors of delay

```bash
python -m src.modeling.train_models_rq3
```

**Outputs**:

- `tables/RQ3_Table6_top_predictors.csv` – Top predictors ranked by importance
- `figures/RQ3_Fig4_relative_importance.pdf` – Bar chart of predictor importance
- `figures/RQ3_Fig5_shap_summary.pdf` – SHAP beeswarm plot
- `figures/RQ3_Fig6_delay_vs_leadtime.pdf` – Delay probability by lead time bucket

All figures and tables are saved in PDF and CSV formats, ready for submission.

***

### 5.5. Optional: Export tables to Excel

If you need Excel versions of the tables for submission:

```bash
python src/evaluation/export_tables_to_excel.py
```

This converts all CSV tables in `tables/` to `.xlsx` format.

***

### 5.6. Complete pipeline summary

To run the entire pipeline from raw data to final outputs:

```bash
# 1. Clean datasets
python -m src.data_ingestion.run_DataCo_pipeline
python -m src.data_ingestion.run_DataSCMS_pipeline

# 2. Merge and engineer features
python -m src.feature_engineering.merge_multisource
python -m src.feature_engineering.engineer_features

# 3. Generate all RQ outputs
python -m src.modeling.train_models_dataco
python -m src.modeling.train_models_rq2
python -m src.modeling.train_models_rq3

# 4. (Optional) Export to Excel
python src/evaluation/export_tables_to_excel.py
```

All outputs will be saved under `figures/` and `tables/`.

***

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

