# Taipei Real Estate Dashboard

A Streamlit dashboard for exploring and predicting residential real-estate
prices in Taipei, built on public transaction data from the Ministry of the
Interior (MOI). The project covers the full pipeline from raw CSV ingestion
to model training and an interactive UI.

## Live Demo

https://aurora-realestate.streamlit.app

## Features

- Cleaning pipeline for MOI real-estate transaction CSVs (handles ROC year,
  Chinese address parsing, outlier filtering)
- SQLite-backed query layer
- Price prediction with Random Forest (R² ≈ 0.56 on held-out set)
- Two-page Streamlit UI: data exploration and price estimation

## Tech Stack

Python 3.11, pandas, scikit-learn, Streamlit, Plotly, SQLite, joblib.

## Project Layout

```
src/
  data_pipeline.py    # ETL: raw CSV -> SQLite
  model.py            # train and persist regression models
app.py                # Streamlit entry point
requirements.txt
data.db               # generated, gitignored
models/               # generated, gitignored
raw_*.csv             # MOI source files, gitignored
```

## Data

Source: Ministry of the Interior, Real Estate Transaction Open Data
(https://plvr.land.moi.gov.tw/DownloadOpenData)

Coverage: Taipei City residential transactions, 2025 Q2 – 2026 Q1,
~22k raw records reduced to ~9.5k after cleaning.

The cleaning step keeps only residential uses (住家用 / 住商用 /
見其他登記事項), parses ROC dates, derives district from the address string,
converts square meters to ping, and drops outliers (unit price outside the
5–95th percentile, area outside 5–200 ping, age outside 0–80 years).

## Running Locally

```bash
git clone https://github.com/aurora-qr/taipei-realestate-dashboard.git
cd taipei-realestate-dashboard
pip install -r requirements.txt
python src/data_pipeline.py    # build data.db
python src/model.py            # train models
streamlit run app.py
```

The MOI quarterly CSVs need to be placed at the project root as
`raw_2025Q2.csv`, `raw_2025Q3.csv`, etc. before running the pipeline.

## Notes

Prices are quoted in 萬元/坪 (10k TWD per ping, gross floor area incl.
common space), consistent with MOI and major real-estate platforms in Taiwan.
