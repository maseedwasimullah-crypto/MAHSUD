"""Feature engineering for aggregate monthly incident-count forecasting.

Only past observations are used to construct features for any given month
(no target leakage). Splits are chronological, never shuffled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import TEST_MONTHS

LAG_PERIODS = (1, 2, 3, 6, 12)
TARGET_COL = "incident_count"


def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned incident rows into a complete monthly time series.

    Months with zero recorded incidents are filled explicitly so the series
    is regular. Zero here means 'no incidents recorded in this dataset for
    that month', which may differ from true incidence where coverage lags.
    """
    dates = pd.to_datetime(df["incident_date"])
    periods = dates.dt.to_period("M")
    grouped = df.assign(period=periods).groupby("period").agg(
        incident_count=("incident_id", "count"),
        fatalities=("fatalities", "sum"),
        injuries=("injuries", "sum"),
    )
    if len(grouped) == 0:
        out = pd.DataFrame(
            columns=["period", "incident_count", "fatalities", "injuries"]
        )
        out["month_start"] = pd.Series(dtype="datetime64[ns]")
        out["year"] = pd.Series(dtype="int64")
        out["month"] = pd.Series(dtype="int64")
        out[TARGET_COL] = pd.Series(dtype="float64")
        return out
    full_index = pd.period_range(periods.min(), periods.max(), freq="M")
    out = grouped.reindex(full_index, fill_value=0)
    out.index.name = "period"
    out = out.reset_index()
    out["month_start"] = out["period"].dt.to_timestamp()
    out["year"] = out["period"].dt.year
    out["month"] = out["period"].dt.month
    out[TARGET_COL] = out["incident_count"].astype(float)
    return out


def add_lag_features(series: pd.DataFrame, lags: tuple[int, ...] = LAG_PERIODS) -> pd.DataFrame:
    sort_col = "month_start" if "month_start" in series.columns else TARGET_COL
    out = series.sort_values(sort_col).reset_index(drop=True).copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[TARGET_COL].shift(lag)
    out["rolling_mean_3"] = out[TARGET_COL].shift(1).rolling(window=3).mean()
    out["rolling_mean_12"] = out[TARGET_COL].shift(1).rolling(window=12).mean()
    out["trend_index"] = np.arange(len(out))
    return out


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dummies = pd.get_dummies(out["month"], prefix="m", dtype=float)
    out = pd.concat([out, dummies], axis=1)
    return out


FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAG_PERIODS]
    + ["rolling_mean_3", "rolling_mean_12", "trend_index"]
    + [f"m_{m}" for m in range(1, 13)]
)


def build_feature_matrix(series: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return feature-ready frame (rows with any NaN in features dropped) and columns."""
    df = add_seasonal_features(add_lag_features(series))
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    usable = df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    usable[feature_cols] = usable[feature_cols].astype(float)
    return usable, feature_cols


def chronological_split(
    feature_df: pd.DataFrame, test_months: int = TEST_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split on the time index (no shuffling)."""
    if len(feature_df) <= test_months + 12:
        raise ValueError(
            f"Insufficient observations ({len(feature_df)}) for a chronological "
            f"split with {test_months} test months and >=12 training months."
        )
    train = feature_df.iloc[:-test_months].copy()
    test = feature_df.iloc[-test_months:].copy()
    if train["month_start"].max() >= test["month_start"].min():
        raise AssertionError("Leakage check failed: train period overlaps test period.")
    return train, test
