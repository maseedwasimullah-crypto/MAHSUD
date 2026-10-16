"""Tests for the MAHSUD modeling pipeline (training, metrics, forecasting)."""
import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean_incident_data
from src.data_collection import generate_demo_dataset
from src.evaluate_model import compare_to_baselines, regression_metrics
from src.feature_engineering import build_feature_matrix, build_monthly_series, chronological_split
from src.forecasting import recursive_forecast
from src.train_model import make_pipeline, train_models


@pytest.fixture(scope="module")
def cleaned_df():
    df = generate_demo_dataset(n_duplicates=10, n_missing_dates=3, casualty_missing_rate=0.05, seed=42)
    cleaned, _ = clean_incident_data(df, report_path=None)
    return cleaned


def test_regression_metrics_values():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 4.0])
    m = regression_metrics(y, p)
    assert m["mae"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["rmse"] == pytest.approx(np.sqrt((0 + 0 + 1) / 3), abs=1e-3)


def test_compare_to_baselines_flags_improvement():
    model = {"mae": 5.0}
    naive = {"mae": 10.0}
    seasonal = {"mae": 12.0}
    cmp = compare_to_baselines(model, naive, seasonal)
    assert cmp["beats_naive"] is True
    assert cmp["mae_improvement_vs_naive_pct"] == pytest.approx(50.0)


def test_pipeline_builders():
    assert make_pipeline("linear_regression") is not None
    assert make_pipeline("random_forest") is not None
    with pytest.raises(ValueError):
        make_pipeline("not_a_model")


def test_train_models_runs_and_beats_or_reports(cleaned_df):
    best, bundle = train_models(cleaned_df, model_names=("linear_regression",), test_months=24)
    assert best.model_name == "linear_regression"
    assert best.n_test == 24
    assert best.n_train > best.n_test
    assert best.metrics_test["mae"] >= 0
    assert best.baseline_naive_test["mae"] > 0
    assert set(bundle["pipelines"]) == {"linear_regression"}
    assert best.test_start < best.test_end


def test_training_fails_gracefully_on_tiny_data():
    df = generate_demo_dataset(n_duplicates=0, n_missing_dates=0, casualty_missing_rate=0.0, seed=3)
    df = df[pd.to_datetime(df["incident_date"]).dt.year == 2013]
    cleaned, _ = clean_incident_data(df, report_path=None)
    with pytest.raises(ValueError):
        train_models(cleaned, model_names=("linear_regression",), test_months=6)


def test_recursive_forecast_shape_and_bounds(cleaned_df):
    best, bundle = train_models(cleaned_df, model_names=("linear_regression",), test_months=18)
    series = build_monthly_series(cleaned_df)
    pipe = bundle["pipelines"]["linear_regression"]
    fc = recursive_forecast(pipe, series, horizon_months=12, residual_std=3.0)
    assert len(fc) == 12
    assert (fc["forecast_count"] >= 0).all()
    assert (fc["lower_95"] <= fc["forecast_count"]).all()
    assert (fc["forecast_count"] <= fc["upper_95"]).all()
    assert fc["month_start"].iloc[0] > series["month_start"].max()


def test_feature_matrix_has_no_nan_and_leakage_safe_order(cleaned_df):
    series = build_monthly_series(cleaned_df)
    feats, cols = build_feature_matrix(series)
    assert feats[cols + ["incident_count"]].isna().sum().sum() == 0
    first = feats.iloc[0]
    assert first["lag_12"] == series.loc[series.index[feats.index[0] - 12], "incident_count"]
