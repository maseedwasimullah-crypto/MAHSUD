"""End-to-end MAHSUD pipeline: data -> clean -> EDA -> train -> forecast -> figures.

Run from the project root:
    python run_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEMO_DATA_PATH,
    EDA_REPORT_PATH,
    MONTHLY_DATA_PATH,
    RAW_DATA_PATH,
)
from src.data_collection import load_incident_data, write_demo_dataset
from src.data_cleaning import clean_incident_data, save_processed
from src.exploratory_analysis import (
    category_distribution,
    district_summary,
    monthly_counts,
    run_full_eda,
    yearly_counts,
)
from src.feature_engineering import build_monthly_series
from src.forecasting import forecast_and_save
from src.train_model import save_artifacts, train_models
from src.visualization import generate_all_figures


def main() -> int:
    print("=" * 70)
    print("MAHSUD pipeline — historical aggregate analysis (research use)")
    print("=" * 70)

    if not RAW_DATA_PATH.exists() and not DEMO_DATA_PATH.exists():
        print("[1/6] No dataset found -> generating SYNTHETIC demonstration dataset...")
        write_demo_dataset()
    df_raw, provenance = load_incident_data()
    print(f"[1/6] Loaded {provenance['rows']} rows from {provenance['file_used']}")
    print(f"      data_type: {provenance['data_type']}")
    if provenance["data_type"] == "synthetic_demonstration":
        print("      *** WARNING: SYNTHETIC demonstration data. Not real incidents. ***")

    print("[2/6] Cleaning...")
    df_clean, clean_report = clean_incident_data(df_raw)
    save_processed(df_clean)
    series = build_monthly_series(df_clean)
    series_save = series.drop(columns=["period"]).copy()
    series_save.to_csv(MONTHLY_DATA_PATH, index=False)
    print(
        f"      {clean_report['rows_received']} -> {clean_report['rows_after_cleaning']} rows; "
        f"{len(series)} monthly observations; saved {MONTHLY_DATA_PATH.name}"
    )

    print("[3/6] Exploratory analysis...")
    is_demo = provenance["data_type"] == "synthetic_demonstration"
    eda = run_full_eda(df_clean, EDA_REPORT_PATH)
    print(f"      EDA summary -> {EDA_REPORT_PATH}")

    print("[4/6] Training models (chronological split, random_state=42)...")
    try:
        best, bundle = train_models(df_clean)
    except ValueError as exc:
        print(f"      Training skipped: {exc}")
        return 1
    save_artifacts(bundle, best)
    print(
        f"      Best model: {best.model_name} | test MAE={best.metrics_test['mae']} "
        f"RMSE={best.metrics_test['rmse']} | naive MAE={best.baseline_naive_test['mae']}"
    )
    comparison = json.loads(Path("reports/model_metrics.json").read_text(encoding="utf-8"))
    from src.evaluate_model import compare_to_baselines

    best_all = comparison["all_models"][best.model_name]
    cmp = compare_to_baselines(
        best_all["metrics_test"],
        best_all["baseline_naive_test"],
        best_all["baseline_seasonal_naive_test"],
    )
    print(f"      Beats naive baseline: {cmp['beats_naive']} | vs seasonal naive: {cmp['beats_seasonal_naive']}")

    print("[5/6] Forecasting (uncertain aggregate estimates)...")
    pipeline = bundle["pipelines"][best.model_name]
    meta = json.loads(Path("models/incident_count_model_meta.json").read_text(encoding="utf-8"))
    forecast = forecast_and_save(pipeline, meta, df_clean)
    print(f"      {len(forecast)} forecast months saved.")

    print("[6/6] Generating figures...")
    test_df = bundle["test_df"]
    test_pred = test_df[["month_start"]].copy()
    test_pred["predicted"] = pipeline.predict(test_df[best.feature_columns])
    paths = generate_all_figures(
        yearly=yearly_counts(df_clean),
        monthly=monthly_counts(df_clean),
        districts=district_summary(df_clean),
        categories=category_distribution(df_clean),
        series=series,
        forecast=forecast,
        test_pred=test_pred,
        demo=is_demo,
    )
    for p in paths:
        print(f"      figure: {p}")

    print("=" * 70)
    print("Pipeline complete. Next: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
