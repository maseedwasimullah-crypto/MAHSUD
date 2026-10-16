"""MAHSUD — Streamlit dashboard for historical KPK incident analysis.

Launch from the project root:
    python -m streamlit run app.py

This dashboard is a research and public-awareness interface. It shows
historical aggregate statistics only. It does not predict specific
attacks, identify suspects, or profile communities.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    FORECAST_PATH,
    METRICS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)
from src.data_cleaning import clean_incident_data, load_processed
from src.data_collection import get_source_registry, load_incident_data, write_demo_dataset
from src.exploratory_analysis import (
    casualty_summary,
    category_distribution,
    data_quality_checks,
    dataset_overview,
    district_summary,
    filter_incidents,
    frequency_change_stats,
    monthly_counts,
    run_full_eda,
    source_coverage,
    yearly_counts,
)
from src.feature_engineering import build_monthly_series
from src.train_model import load_model, save_artifacts, train_models
from src.visualization import (
    plotly_districts,
    plotly_forecast,
    plotly_monthly,
    plotly_yearly,
)

st.set_page_config(
    page_title="MAHSUD — KPK Incident Analysis",
    page_icon="📊",
    layout="wide",
)

ETHICS_NOTE = (
    "MAHSUD is a research and public-awareness tool. It analyses historical, "
    "aggregate statistics only. It must NOT be used to predict specific attacks, "
    "target individuals or communities, infer that any group is dangerous, or "
    "support operational/security targeting. Differences between districts may "
    "reflect reporting coverage as much as real-world variation."
)


@st.cache_data(show_spinner="Loading and cleaning incident data...")
def load_data():
    if not RAW_DATA_PATH.exists() and not Path("data/sample/kpk_incidents_demo.csv").exists():
        write_demo_dataset()
    raw, provenance = load_incident_data()
    cleaned, clean_report = clean_incident_data(raw, report_path=None)
    return cleaned, provenance, clean_report


@st.cache_resource(show_spinner="Loading saved model (trusted local files only)...")
def get_saved_model():
    if MODEL_PATH.exists() and MODEL_META_PATH.exists():
        return load_model()
    return None, None


def demo_banner(provenance: dict):
    if provenance["data_type"] == "synthetic_demonstration":
        st.error(
            "**SYNTHETIC DEMONSTRATION DATA IN USE.** Every incident row is computer-generated "
            "for testing the software. None of these records are real events. Replace with "
            "verified data (see *Sources & Methodology*) before drawing any real-world conclusion."
        )


def last_update_caption(cleaned: pd.DataFrame):
    if len(cleaned):
        latest = pd.to_datetime(cleaned["incident_date"]).max().date()
        st.caption(f"Latest incident date present in loaded dataset: **{latest.isoformat()}**")
    if RAW_DATA_PATH.exists():
        mtime = datetime.fromtimestamp(RAW_DATA_PATH.stat().st_mtime).date()
        st.caption(f"Raw data file last modified on this machine: **{mtime.isoformat()}**")


def sidebar_filters(cleaned: pd.DataFrame):
    st.sidebar.header("Filters")
    d_min = pd.to_datetime(cleaned["incident_date"]).min().date()
    d_max = pd.to_datetime(cleaned["incident_date"]).max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(d_min, d_max), min_value=d_min, max_value=d_max
    )
    districts = sorted(cleaned["district"].unique().tolist())
    selected_districts = st.sidebar.multiselect("Districts (empty = all)", districts)
    categories = sorted(cleaned["incident_category"].unique().tolist())
    selected_categories = st.sidebar.multiselect("Incident categories (empty = all)", categories)
    start = end = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        start = end = date_range[0]
    filtered = filter_incidents(cleaned, start, end, selected_districts, selected_categories)
    st.sidebar.metric("Incidents in filter", len(filtered))
    return filtered


def page_home(provenance: dict):
    st.title("MAHSUD — KPK Terrorism Incident Analysis")
    st.subheader("Historical trends, statistics, and uncertain aggregate forecasts")
    st.markdown(
        """
**What this tool does**

* Loads a documented incident dataset for Khyber Pakhtunkhwa, Pakistan
* Cleans and validates records with an auditable report
* Analyses yearly/monthly trends, casualties, districts, and categories
* Trains scikit-learn models on aggregate monthly counts with a simple baseline comparison
* Produces uncertain aggregate forecasts with rough uncertainty intervals
* Visualises everything in an interactive dashboard

**What this tool deliberately does NOT do**

* Predict the time, place, or method of future attacks
* Identify or list suspects
* Claim any district, ethnic group, religious group, or individual is inherently dangerous
* Provide operational or tactical security guidance
        """
    )
    st.warning(ETHICS_NOTE)
    demo_banner(provenance)
    last_update_caption(load_data()[0])
    col1, col2, col3 = st.columns(3)
    cleaned, _, _ = load_data()
    with col1:
        st.metric("Recorded incidents loaded", f"{len(cleaned):,}")
    with col2:
        st.metric("Districts", cleaned["district"].nunique())
    with col3:
        span = (
            f"{pd.to_datetime(cleaned['incident_date']).min().date()} → "
            f"{pd.to_datetime(cleaned['incident_date']).max().date()}"
        )
        st.metric("Date span", span)


def page_overview(filtered: pd.DataFrame, provenance: dict):
    st.header("Data Overview")
    demo_banner(provenance)
    overview = dataset_overview(filtered)
    casualties = casualty_summary(filtered)
    quality = data_quality_checks(filtered)
    freq = frequency_change_stats(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded incidents", f"{overview['total_recorded_incidents']:,}")
    c2.metric("Recorded fatalities (sum)", f"{overview['recorded_fatalities_sum']:,.0f}")
    c3.metric("Recorded injuries (sum)", f"{overview['recorded_injuries_sum']:,.0f}")
    c4.metric("Distinct districts", overview["distinct_districts"])

    st.info(overview["caveat"])
    tab1, tab2, tab3 = st.tabs(["Casualty detail", "Data quality", "Frequency change"])
    with tab1:
        st.json(casualties)
    with tab2:
        st.json(quality)
    with tab3:
        st.json(freq)
    with st.expander("Monthly aggregate table (used by the ML pipeline)"):
        series = build_monthly_series(filtered)
        st.dataframe(series.drop(columns=["period"]), width='stretch', hide_index=True)


def page_trends(filtered: pd.DataFrame, provenance: dict):
    st.header("Historical Trend Analysis")
    demo_banner(provenance)
    is_demo = provenance["data_type"] == "synthetic_demonstration"
    yearly = yearly_counts(filtered)
    monthly = monthly_counts(filtered)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plotly_yearly(yearly, demo=is_demo), width='stretch')
    with col2:
        st.plotly_chart(plotly_monthly(monthly, demo=is_demo), width='stretch')
    st.subheader("Recorded casualties by year")
    st.bar_chart(yearly.set_index("year")[["fatalities", "injuries"]])
    st.caption(
        "Casualty sums exclude rows where the field was reported as missing; "
        "missing is not treated as zero."
    )


def page_districts(filtered: pd.DataFrame, provenance: dict):
    st.header("District-Level Aggregate Statistics")
    demo_banner(provenance)
    st.warning(
        "District comparisons describe **recorded incidents in this dataset only**. "
        "They do not measure the character of a district or its residents. Reporting "
        "coverage differs across districts and years."
    )
    districts = district_summary(filtered)
    st.plotly_chart(plotly_districts(districts, demo=provenance["data_type"] == "synthetic_demonstration"), width='stretch')
    st.dataframe(districts, width='stretch', hide_index=True)
    st.subheader("Category distribution")
    st.dataframe(category_distribution(filtered), width='stretch', hide_index=True)
    st.subheader("Source coverage")
    st.dataframe(source_coverage(filtered), width='stretch', hide_index=True)


def page_model(filtered: pd.DataFrame, provenance: dict):
    st.header("Model Training & Evaluation")
    demo_banner(provenance)
    st.markdown(
        """
**Task:** estimate next-month **aggregate recorded incident counts** for the whole loaded
dataset, using lagged historical counts and calendar features.

* Split: **chronological** (train on the past, test on the most recent months) — no shuffling
* Baselines: naive last-value and seasonal naive (same month, prior year)
* Models: Linear Regression (scaled features) and Random Forest
* Metrics: MAE and RMSE on the held-out recent window
* `random_state = 42` everywhere for reproducibility

These are uncertain statistical estimates of aggregate counts — **not predictions of
specific attacks**.
        """
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        retrain = st.button("Retrain models now", type="primary")
    if retrain:
        with st.spinner("Training (this may take a minute)..."):
            try:
                best, bundle = train_models(filtered)
                save_artifacts(bundle, best)
                st.success(f"Trained. Best model: **{best.model_name}**")
                st.cache_resource.clear()
            except ValueError as exc:
                st.error(f"Training not possible: {exc}")
                return

    if not METRICS_PATH.exists():
        st.warning(
            "No saved model metrics found. Click *Retrain models now* or run "
            "`python run_pipeline.py` in the terminal."
        )
        return

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = metrics["best_model"]
    st.subheader("Best model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", best["model_name"].replace("_", " ").title())
    m2.metric("Test MAE", f"{best['metrics_test']['mae']:.2f}")
    m3.metric("Test RMSE", f"{best['metrics_test']['rmse']:.2f}")
    naive_mae = best["baseline_naive_test"]["mae"]
    m4.metric("Naive baseline MAE", f"{naive_mae:.2f}")
    improvement = (naive_mae - best["metrics_test"]["mae"]) / max(naive_mae, 1e-9) * 100
    if improvement > 0:
        st.success(
            f"The model reduces MAE by **{improvement:.1f}%** versus the naive baseline "
            f"on the held-out window ({best['test_start']} to {best['test_end']}, "
            f"n={best['n_test']} months)."
        )
    else:
        st.warning(
            "The model does **not** beat the naive baseline on the held-out window. "
            "Treat any model-based figures with caution; the baseline may be preferable."
        )
    st.subheader("Full comparison (all trained models)")
    st.dataframe(pd.DataFrame(metrics["all_models"]).T, width='stretch')
    if MODEL_META_PATH.exists():
        with st.expander("Model metadata / load policy"):
            st.json(json.loads(MODEL_META_PATH.read_text(encoding="utf-8")))


def page_forecast(provenance: dict):
    st.header("Uncertain Aggregate Forecast")
    demo_banner(provenance)
    st.warning(
        "Forecasts assume past recorded patterns continue. They are **uncertain estimates "
        "of aggregate monthly counts**, not predictions of specific attacks, dates, or "
        "locations. Intervals are rough residual-based bounds, not formal prediction intervals."
    )
    if not (FORECAST_PATH.exists() and MODEL_PATH.exists()):
        st.info("No forecast artifacts yet. Run `python run_pipeline.py` or retrain on the Model page.")
        return
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["month_start"])
    cleaned, _, _ = load_data()
    series = build_monthly_series(filter_incidents(cleaned))
    st.plotly_chart(plotly_forecast(series, forecast, demo=provenance["data_type"] == "synthetic_demonstration"), width='stretch')
    st.subheader("Forecast table")
    st.dataframe(forecast.drop(columns=["interval_note"]), width='stretch', hide_index=True)
    st.caption(forecast["interval_note"].iloc[0] if len(forecast) else "")


def page_sources(provenance: dict, clean_report: dict):
    st.header("Data Sources & Methodology")
    demo_banner(provenance)
    st.subheader("Dataset currently loaded")
    st.json(provenance)
    st.subheader("Cleaning decisions (auditable)")
    st.json(clean_report)
    st.subheader("Candidate public sources for verified data")
    st.markdown(
        "Place a verified CSV at `data/raw/kpk_incidents.csv` using the same column names; "
        "it automatically takes priority over the demonstration file."
    )
    st.dataframe(pd.DataFrame(get_source_registry()), width='stretch')
    st.subheader("Methodology summary")
    st.markdown(
        """
1. **Load** raw CSV (`data/raw/...` if present, else the synthetic demo file)
2. **Clean**: parse dates (drop unparseable), normalize districts, drop ID and content
   duplicates, coerce casualties to numeric while keeping missing as NaN, derive year/month
3. **Aggregate**: complete monthly series (zero-filled months = no *recorded* incidents)
4. **Features**: lags 1/2/3/6/12, rolling means, trend index, month dummies — built only
   from past values (no leakage)
5. **Split**: chronological hold-out of the most recent months
6. **Train**: Linear Regression and Random Forest vs naive and seasonal-naive baselines
7. **Evaluate**: MAE / RMSE on the hold-out window
8. **Forecast**: recursive multi-step estimates with rough residual-based intervals
        """
    )
    st.subheader("Responsible-use statement")
    st.error(ETHICS_NOTE)


def main():
    cleaned, provenance, clean_report = load_data()
    filtered = sidebar_filters(cleaned)

    pages = {
        "Home": lambda: page_home(provenance),
        "Data Overview": lambda: page_overview(filtered, provenance),
        "Historical Trends": lambda: page_trends(filtered, provenance),
        "District Analysis": lambda: page_districts(filtered, provenance),
        "Model Training & Evaluation": lambda: page_model(filtered, provenance),
        "Forecasting": lambda: page_forecast(provenance),
        "Sources & Methodology": lambda: page_sources(provenance, clean_report),
    }
    choice = st.sidebar.radio("Navigate", list(pages.keys()))
    pages_requiring_data = {
        "Data Overview",
        "Historical Trends",
        "District Analysis",
        "Model Training & Evaluation",
    }
    if len(filtered) == 0 and choice in pages_requiring_data:
        st.header(choice)
        st.warning(
            "No incidents match the current sidebar filters. Adjust the date range, "
            "districts, or categories to see results."
        )
    else:
        pages[choice]()
    st.sidebar.divider()
    st.sidebar.caption(
        "Responsible-use tool: historical aggregate research only. "
        "No attack predictions, no suspect identification, no community profiling."
    )


if __name__ == "__main__":
    main()
