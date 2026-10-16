"""Central configuration for the MAHSUD project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = RAW_DIR / "kpk_incidents.csv"
DEMO_DATA_PATH = SAMPLE_DIR / "kpk_incidents_demo.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "kpk_incidents_clean.csv"
MONTHLY_DATA_PATH = PROCESSED_DIR / "kpk_monthly_aggregates.csv"

MODEL_PATH = MODELS_DIR / "incident_count_model.joblib"
MODEL_META_PATH = MODELS_DIR / "incident_count_model_meta.json"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
CLEANING_REPORT_PATH = REPORTS_DIR / "data_cleaning_report.json"
EDA_REPORT_PATH = REPORTS_DIR / "eda_summary.json"
FORECAST_PATH = PROCESSED_DIR / "forecast_monthly.csv"

RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "incident_id",
    "incident_date",
    "year",
    "month",
    "district",
    "province",
    "incident_category",
    "fatalities",
    "injuries",
    "source_name",
    "source_url",
    "verification_status",
]

PROVINCE_NAME = "Khyber Pakhtunkhwa"

FORECAST_HORIZON_MONTHS = 24
TEST_MONTHS = 24

DASHBOARD_LAST_UPDATED_KEY = "dataset_last_verified_update"
