"""Model training for MAHSUD: baseline vs scikit-learn regressors.

Forecast target: aggregate recorded incident COUNT per month for KPK.
This is an uncertain statistical estimate of recorded-incident volume,
never a prediction of specific attacks, times, or places.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    METRICS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_MONTHS,
)
from src.evaluate_model import regression_metrics
from src.feature_engineering import (
    TARGET_COL,
    build_feature_matrix,
    build_monthly_series,
    chronological_split,
)


@dataclass
class TrainResult:
    model_name: str
    metrics_test: dict
    metrics_train: dict
    baseline_naive_test: dict
    baseline_seasonal_naive_test: dict
    n_train: int
    n_test: int
    test_start: str
    test_end: str
    feature_columns: list
    trained_at_utc: str


def build_baseline_predictions(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Naive (last train value) and seasonal naive (same month, prior year)."""
    naive_value = float(train[TARGET_COL].iloc[-1])
    naive_preds = np.full(len(test), naive_value)
    seasonal = {}
    for _, row in train.iterrows():
        seasonal[int(row["month"])] = float(row[TARGET_COL])
    seasonal_preds = np.array(
        [seasonal.get(int(m), naive_value) for m in test["month"]], dtype=float
    )
    return naive_preds, seasonal_preds


def make_pipeline(model_name: str) -> Pipeline:
    if model_name == "linear_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                )
            ]
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def train_models(
    df_clean: pd.DataFrame,
    model_names: tuple[str, ...] = ("linear_regression", "random_forest"),
    test_months: int = TEST_MONTHS,
) -> tuple[TrainResult, dict]:
    """Train baselines + supervised models on monthly aggregates.

    Returns the best result (lowest test MAE among supervised models) and a
    dict with all fitted pipelines keyed by model name.
    """
    series = build_monthly_series(df_clean)
    feature_df, feature_cols = build_feature_matrix(series)
    if len(feature_df) < 24:
        raise ValueError(
            f"Only {len(feature_df)} usable monthly observations after lagging; "
            "need at least 24 to train a meaningful model. Obtain more history "
            "or reduce the lag configuration."
        )
    split_test = min(test_months, max(6, len(feature_df) // 5))
    train, test = chronological_split(feature_df, test_months=split_test)

    y_train = train[TARGET_COL].to_numpy(dtype=float)
    y_test = test[TARGET_COL].to_numpy(dtype=float)
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    naive_preds, seasonal_preds = build_baseline_predictions(train, test)
    fitted: dict[str, Pipeline] = {}
    results: list[TrainResult] = []
    now = datetime.now(timezone.utc).isoformat()

    for name in model_names:
        pipe = make_pipeline(name)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred_test = pipe.predict(X_test)
        pred_train = pipe.predict(X_train)
        results.append(
            TrainResult(
                model_name=name,
                metrics_test=regression_metrics(y_test, pred_test),
                metrics_train=regression_metrics(y_train, pred_train),
                baseline_naive_test=regression_metrics(y_test, naive_preds),
                baseline_seasonal_naive_test=regression_metrics(y_test, seasonal_preds),
                n_train=len(train),
                n_test=len(test),
                test_start=str(test["month_start"].min().date()),
                test_end=str(test["month_start"].max().date()),
                feature_columns=feature_cols,
                trained_at_utc=now,
            )
        )

    best = min(results, key=lambda r: r.metrics_test["mae"])
    bundle = {
        "pipelines": fitted,
        "all_results": {r.model_name: asdict(r) for r in results},
        "best_model_name": best.model_name,
        "series": series,
        "train_df": train,
        "test_df": test,
        "y_test": y_test,
        "naive_preds": naive_preds,
        "seasonal_preds": seasonal_preds,
    }
    return best, bundle


def save_artifacts(
    bundle: dict,
    best: TrainResult,
    model_path: Path = MODEL_PATH,
    meta_path: Path = MODEL_META_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Persist the best pipeline, metadata, and full comparison metrics."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = bundle["pipelines"][best.model_name]
    joblib.dump(pipeline, model_path, compress=3)

    residual = bundle["y_test"] - bundle["pipelines"][best.model_name].predict(
        bundle["test_df"][best.feature_columns]
    )
    meta = {
        "model_name": best.model_name,
        "target": "monthly recorded incident count (aggregate)",
        "feature_columns": best.feature_columns,
        "trained_at_utc": best.trained_at_utc,
        "n_train": best.n_train,
        "n_test": best.n_test,
        "test_start": best.test_start,
        "test_end": best.test_end,
        "random_state": RANDOM_STATE,
        "model_file": str(model_path),
        "test_residual_std": float(np.std(residual, ddof=1)),
        "safety_note": (
            "Aggregate historical estimates only. Not a prediction of specific "
            "attacks, locations, or perpetrators."
        ),
        "load_policy": (
            "Only load model files produced by this pipeline on this machine. "
            "Never load untrusted joblib/pickle files."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_model": asdict(best),
        "all_models": bundle["all_results"],
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(model_path: Path = MODEL_PATH, meta_path: Path = MODEL_META_PATH):
    """Load a previously trained pipeline plus its metadata.

    Safety: only call this for files written by this project on this machine.
    """
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "No saved model found. Run: python -m src.train_model"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(model_path)
    return pipeline, meta
