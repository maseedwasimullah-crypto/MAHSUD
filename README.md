# MAHSUD — KPK Terrorism Incident Analysis AI

MAHSUD is a Python data-analysis and Machine Learning application that studies
**historical, aggregate** terrorism-incident statistics for Khyber Pakhtunkhwa
(KPK), Pakistan. It cleans documented incident data, explores trends, trains
forecasting models on aggregate monthly counts, and presents everything in a
Streamlit dashboard.

> **Responsible-use statement.** MAHSUD is a research and public-awareness tool.
> It analyses recorded historical aggregates only. It does **not** predict
> specific attacks, identify suspects, profile communities, or provide
> operational security guidance. District-level differences in the dataset may
> reflect reporting coverage as much as real-world variation and must never be
> read as claims that any place or group is inherently dangerous.

---

## 1. Project objectives

- Analyse historical terrorism incidents in KPK (and recent data when verified sources are available)
- Identify changes in recorded incident frequency and casualties over time
- Compare aggregate statistics across districts and categories
- Visualise monthly/yearly trends with clear caveats
- Train and evaluate ML models on aggregate monthly counts against simple baselines
- Provide a beginner-friendly dashboard for exploring results

## 2. Requirements

- **Python 3.10+** (developed and tested on Python 3.14, Windows)
- Windows, macOS, or Linux (documentation uses Windows PowerShell)
- ~500 MB free disk for dependencies

## 3. Installation (Windows PowerShell)

Open VS Code, open a terminal (`Ctrl` + `` ` ``), then:

```powershell
# from the project root (the folder containing app.py)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation scripts (one-time):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 4. Dataset: how to obtain and prepare

### 4.1 What ships with the project

The project **ships with a clearly labeled SYNTHETIC demonstration dataset**
(`data/sample/kpk_incidents_demo.csv`). Every row has
`verification_status = "demonstration_synthetic"`. **These are not real
incidents** and must never be cited as such. They exist so the full pipeline
can run offline.

See `data/sample/DATASET_PROVENANCE.md` for full provenance documentation.

### 4.2 Using verified data instead

1. Obtain data from a documented public source (see the *Sources & Methodology*
   page in the dashboard, or `src/data_collection.py::PUBLIC_SOURCES`):
   - Pakistan Institute for Peace Studies (PIPS) — https://www.ips.org.pk/
   - SATP — respect https://www.satp.org/disclaimers
   - NACTA — https://nacta.gov.pk/
   - START Global Terrorism Database — https://www.start.umd.edu/gtd/
2. Format as CSV with these columns:

   | Column | Notes |
   | --- | --- |
   | incident_id | unique identifier |
   | incident_date | ISO date `YYYY-MM-DD` |
   | district | KPK district name |
   | province | e.g. `Khyber Pakhtunkhwa` |
   | incident_category | e.g. `armed_attack`, `explosive_device` |
   | fatalities | integer; leave blank if unknown (**do not invent zeros**) |
   | injuries | integer; leave blank if unknown |
   | source_name | citation |
   | source_url | link |
   | verification_status | honest label per row |

3. Save as `data/raw/kpk_incidents.csv` — it automatically takes priority over
   the demonstration file.
4. Document the source (name, URL, retrieval date, coverage, limitations) in
   `data/sample/DATASET_PROVENANCE.md` and the dashboard Sources page.

## 5. Running the analysis pipeline

```powershell
python run_pipeline.py
```

This will (re)generate data if missing, clean it, compute EDA summaries, train
models, forecast 24 months, and write figures. Outputs:

- `data/processed/kpk_incidents_clean.csv` — cleaned incident rows
- `data/processed/kpk_monthly_aggregates.csv` — monthly time series
- `data/processed/forecast_monthly.csv` — uncertain aggregate forecasts
- `reports/data_cleaning_report.json` — auditable cleaning decisions
- `reports/eda_summary.json` — statistics
- `reports/model_metrics.json` — model vs baseline comparison
- `reports/figures/*.png` — charts
- `models/incident_count_model.joblib` + `models/incident_count_model_meta.json`

You can also regenerate only the demo dataset:

```powershell
python -m src.data_collection
```

## 6. Training the model

Training happens inside `python run_pipeline.py`, or from the dashboard's
*Model Training & Evaluation* page (button: *Retrain models now*).

Design:

- **Target:** monthly recorded incident count (aggregate)
- **Features:** lags 1/2/3/6/12, 3- and 12-month rolling means, trend index,
  month dummies — all built strictly from past values (no leakage)
- **Split:** chronological hold-out of the most recent 24 months (never shuffled)
- **Models:** Linear Regression (standardized) and Random Forest (300 trees)
- **Baselines:** naive last-value, seasonal naive (same month, prior year)
- **Metrics:** MAE, RMSE on the held-out window
- **Reproducibility:** `random_state = 42` for generation, splits, and forests
- **Serialization:** `joblib` — only load model files written by this project on
  this machine; **never load untrusted pickle/joblib files**

Example results on the synthetic demonstration dataset (your run should match):

| Model | Test MAE | Test RMSE | Naive baseline MAE | Seasonal naive MAE |
| --- | --- | --- | --- | --- |
| Linear Regression | 6.19 | 7.49 | 16.67 | 15.33 |
| Random Forest | 7.47 | 9.05 | 16.67 | 15.33 |

Best model: **Linear Regression** (test window 2023-01 → 2024-12, n_test = 24 months).
These numbers describe **synthetic data** and say nothing about the real world.

## 7. Launching the dashboard

```powershell
python -m streamlit run app.py
```

Sections: Home · Data Overview · Historical Trends · District Analysis ·
Model Training & Evaluation · Forecasting · Sources & Methodology.
Sidebar filters: date range, districts, categories. The app displays a prominent
warning whenever the synthetic dataset is in use.

## 8. Running tests

```powershell
python -m pytest tests -v
```

18 tests cover dataset loading, date parsing, missing-value handling, duplicate
detection, aggregation, feature generation, chronological splitting, model
training, metric calculations, forecasting bounds, and empty/insufficient data.

## 9. Model evaluation methodology

- Chronological (not random) train/test split prevents future-to-past leakage.
- The model must be compared against both baselines; if it fails to beat the
  naive baseline, the dashboard says so explicitly rather than hiding it.
- Forecast uncertainty is a **rough residual-based interval** from test-set
  residuals — not a formal statistical prediction interval.
- All forecasts are labeled as uncertain aggregate estimates, never as
  predictions of specific attacks.

## 10. Known limitations

- **Demonstration data:** unless replaced, all numbers are synthetic.
- **Recorded ≠ actual:** counts reflect what sources recorded; reporting
  coverage varies by year and district.
- **Aggregate forecasting only:** the model estimates next-month *counts*, not
  locations, methods, or perpetrators.
- **Rough intervals:** uncertainty bands are approximate.
- **Manual source ingestion:** verified datasets must be exported manually from
  PIPS/SATP/GTD/NACTA per their terms.
- **Category taxonomy** is coarse; fine-grained classifications depend on the
  source dataset.

## 11. Ethical and responsible use

Do **not** use MAHSUD to:

- predict the time, place, or method of future attacks;
- recommend targeting of individuals, communities, or facilities;
- infer that a district, ethnicity, religion, or person is dangerous;
- compile suspect lists from unverified sources;
- exploit security weaknesses.

District comparisons describe **recorded dataset contents only**. Missing
casualty figures are never treated as zero. Do not cite synthetic rows.

## 12. Project structure

```
MAHSUD/
  app.py                     # Streamlit dashboard
  run_pipeline.py            # end-to-end CLI pipeline
  config.py                  # paths and constants
  requirements.txt
  README.md
  data/
    raw/                     # put verified data here (kpk_incidents.csv)
    processed/               # cleaned data, monthly series, forecasts
    sample/                  # synthetic demo dataset + provenance docs
  models/                    # saved joblib model + metadata
  reports/
    figures/                 # PNG charts
    tables/
    *.json                   # cleaning report, EDA summary, metrics
  src/
    data_collection.py       # source registry + demo dataset generator
    data_cleaning.py         # auditable cleaning pipeline
    exploratory_analysis.py  # statistics
    feature_engineering.py   # monthly series, lags, chronological split
    train_model.py           # baselines + model training + artifacts
    evaluate_model.py        # MAE/RMSE + baseline comparison
    forecasting.py           # recursive aggregate forecast + intervals
    visualization.py         # matplotlib + plotly charts
  tests/
    test_data_cleaning.py
    test_model.py
```

## 13. Troubleshooting

- **`streamlit` not found** — activate the virtual environment
  (`.venv\Scripts\Activate.ps1`) and reinstall `pip install -r requirements.txt`.
- **Matplotlib font-cache message** on first run — harmless, one-time.
- **Training error: insufficient observations** — the dataset covers too few
  months after lagging (need ≥ 24 usable monthly observations). Supply a longer
  date range.
- **Want to reset artifacts** — delete `data/processed/`, `models/`, and
  `reports/`, then run `python run_pipeline.py` again.
