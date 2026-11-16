# Predictive Maintenance: Data Pipeline & Exploratory Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

A small data pipeline and exploratory analysis for machine sensor data of an industrial facility. The goal is to understand which operating conditions are linked to machine failures, as a basis for predictive maintenance.

## What it does

**1. ETL into SQLite (`load_to_sqlite.py`)**
- Reads the raw sensor CSV with pandas
- Renames columns to a clean schema (`AirTemp`, `ProcTemp`, `RPM`, `Torque`, `ToolWear`, `FailureType`)
- Removes duplicate `UDI` records
- Creates the table `predictive_maintenance` in `maintenance.db` (primary key `UDI`) and loads the data with `INSERT OR IGNORE`

**2. Exploratory analysis (`eda.py`)**
- Data overview, summary statistics and missing-value check
- Distribution of failure types
- Correlation matrix of the five sensor variables (air temperature, process temperature, rotational speed, torque, tool wear)
- Temperatures with vs. without failure (boxplots)
- Tool wear by failure type
- Failure rate per machine type

All charts are saved to a `results/` folder.

## Data

The scripts expect the AI4I-style predictive maintenance dataset with these columns:

`UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Failure Type`

The dataset is not included in this repository. Place it at `data/predictive_maintenance.csv`.

## Run it yourself

```bash
pip install -r requirements.txt
mkdir data   # then add predictive_maintenance.csv
python load_to_sqlite.py
python eda.py
```

## Tech stack

Python, SQL (SQLite), pandas, NumPy, Matplotlib, Seaborn

## Possible next steps

- Train a classification model to predict failures (for example Random Forest) and evaluate it with recall and PR-AUC, since failures are rare
- Replace the row-by-row insert with a bulk insert (`DataFrame.to_sql`)
