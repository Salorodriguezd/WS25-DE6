
**Feature engineering:**

build merged DataCo + SCMS dataset.

- Input:

  `data/DataCo_clean_dates_ddmmyyyy.csv`

  `data/SCMS_cleaned.csv`


- Output:
 
  `data/merged/merged_dataco_scms_country_level.csv`


**Note:**

Because the two datasets come from different domains, many rows

may not find a matching SCMS country and SCMS-based features can

be NaN. The goal here is to show a clear, reproducible merging

process between two cleaned sources.



