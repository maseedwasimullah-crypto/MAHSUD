"""Uncertain aggregate forecasting for MAHSUD.

Forecasts are rough statistical estimates of RECORDED monthly incident
counts under the assumption that past patterns continue. They are NOT
predictions of specific attacks and must never be used for targeting,
tactical planning, or claims about individuals or communities.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import FORECAST_HORIZON_MONTHS, FORECAST_PATH, RANDOM_STATE
from src.feature_engineering import (
    FEATURE_COLUMNS,
    LAG_PERIODS,
    TARGET_COL,
    add_lag_features,
    add_seasonal_features,
    build_monthly_series,
)


def recursive_forecast(
    pipeline,
    series: pd.DataFrame,
    horizon_months: int = FORECAST_HORIZON_MONTHS,
    residual_std: float = 0.0,
    confidence_z: float = 1.96,
) -> pd.DataFrame:
    """Forecast future monthly counts recursively from the last known values.

    Uncertainty intervals use the residual standard deviation observed on the
    chronological test set, a rough (not formal prediction-interval) estimate.
    """
    if horizon_months < 1:
        raise ValueError("horizon_months must be >= 1")
    hist = series.sort_values("month_start").reset_index(drop=True).copy()
    last_period = hist["period"].iloc[-1]
    future_periods = pd.period_range(last_period + 1, periods=horizon_months, freq="M")

    working = hist.copy()
    rows = []
    for period in future_periods:
        feats = add_seasonal_features(add_lag_features(working))
        feature_cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
        missing_needed = [c for c in FEATURE_COLUMNS if c not in feats.columns]
        if missing_needed:
            raise ValueError(f"Missing engineered features: {missing_needed}")
        tail = feats.iloc[[-1]]
        if tail[feature_cols].isna().any(axis=None):
            raise ValueError("Feature window insufficient to forecast; need >= 12 months history.")
        pred = float(pipeline.predict(tail[feature_cols].astype(float))[0])
        pred = max(pred, 0.0)
        lower = max(pred - confidence_z * residual_std, 0.0)
        upper = pred + confidence_z * residual_std
        rows.append(
            {
                "period": str(period),
                "month_start": period.to_timestamp(),
                "forecast_count": round(pred, 2),
                "lower_95": round(lower, 2),
                "upper_95": round(upper, 2),
                "interval_note": "rough residual-based interval, not a formal prediction interval",
            }
        )
        working = pd.concat(
            [
                working,
                pd.DataFrame(
                    [
                        {
                            "period": period,
                            "month_start": period.to_timestamp(),
                            "year": period.year,
                            "month": period.month,
                            "incident_count": pred,
                            TARGET_COL: pred,
                            "fatalities": 0.0,
                            "injuries": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    out = pd.DataFrame(rows)
    return out


def forecast_and_save(
    pipeline,
    meta: dict,
    df_clean: pd.DataFrame,
    horizon_months: int = FORECAST_HORIZON_MONTHS,
    out_path: Path = FORECAST_PATH,
) -> pd.DataFrame:
    series = build_monthly_series(df_clean)
    residual_std = float(meta.get("test_residual_std", 0.0))
    fc = recursive_forecast(pipeline, series, horizon_months, residual_std)
    fc.attrs["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    fc.attrs["random_state"] = RANDOM_STATE
    fc.attrs["safety_note"] = (
        "Uncertain aggregate estimates of recorded incident counts. "
        "Not predictions of specific attacks."
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = fc.copy()
        to_save["month_start"] = to_save["month_start"].dt.strftime("%Y-%m-%d")
        to_save.to_csv(out_path, index=False)
    return fc
