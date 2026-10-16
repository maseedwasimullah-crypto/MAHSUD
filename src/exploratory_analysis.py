"""Exploratory statistical analysis for MAHSUD.

All functions operate on historical, aggregated data. Results describe
recorded incidents in the loaded dataset only; they do not measure the
true rate of violence where reporting coverage varies, and they must not
be read as claims that any place or group is inherently dangerous.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import EDA_REPORT_PATH


def filter_incidents(
    df: pd.DataFrame,
    date_start=None,
    date_end=None,
    districts: list[str] | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Filter incidents by optional date range, districts, and categories."""
    out = df
    if date_start is not None:
        out = out[out["incident_date"] >= pd.to_datetime(date_start)]
    if date_end is not None:
        out = out[out["incident_date"] <= pd.to_datetime(date_end)]
    if districts:
        out = out[out["district"].isin(districts)]
    if categories:
        out = out[out["incident_category"].isin(categories)]
    return out.copy()


def dataset_overview(df: pd.DataFrame) -> dict:
    """High-level totals, coverage window, and basic honesty checks."""
    dates = pd.to_datetime(df["incident_date"])
    is_demo = bool(
        (df["verification_status"].astype(str) == "demonstration_synthetic").all()
    ) if len(df) else False
    return {
        "total_recorded_incidents": int(len(df)),
        "date_min": str(dates.min().date()) if len(df) else None,
        "date_max": str(dates.max().date()) if len(df) else None,
        "distinct_districts": int(df["district"].nunique()),
        "distinct_categories": int(df["incident_category"].nunique()),
        "recorded_fatalities_sum": float(df["fatalities"].sum(skipna=True)),
        "recorded_injuries_sum": float(df["injuries"].sum(skipna=True)),
        "fatalities_missing_n": int(df["fatalities"].isna().sum()),
        "injuries_missing_n": int(df["injuries"].isna().sum()),
        "data_is_demonstration_synthetic": is_demo,
        "caveat": (
            "These are counts of RECORDED incidents in this dataset, not estimates "
            "of all incidents that occurred. Reporting coverage varies over time "
            "and across districts."
        ),
    }


def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("year", as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            fatalities=("fatalities", "sum"),
            injuries=("injuries", "sum"),
        )
        .sort_values("year")
    )
    return g


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["period"] = tmp["incident_date"].dt.to_period("M").astype(str)
    g = (
        tmp.groupby("period", as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            fatalities=("fatalities", "sum"),
            injuries=("injuries", "sum"),
        )
        .sort_values("period")
    )
    return g


def district_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = max(len(df), 1)
    g = (
        df.groupby("district", as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            fatalities=("fatalities", "sum"),
            injuries=("injuries", "sum"),
            first_recorded=("incident_date", "min"),
            last_recorded=("incident_date", "max"),
        )
        .sort_values("incidents", ascending=False)
    )
    g["share_of_recorded_incidents_pct"] = (g["incidents"] / total * 100).round(2)
    return g.reset_index(drop=True)


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total = max(len(df), 1)
    g = (
        df.groupby("incident_category", as_index=False)
        .agg(incidents=("incident_id", "count"), fatalities=("fatalities", "sum"))
        .sort_values("incidents", ascending=False)
    )
    g["share_pct"] = (g["incidents"] / total * 100).round(2)
    return g.reset_index(drop=True)


def casualty_summary(df: pd.DataFrame) -> dict:
    fat = df["fatalities"]
    inj = df["injuries"]
    return {
        "fatalities_reported_n": int(fat.notna().sum()),
        "fatalities_missing_n": int(fat.isna().sum()),
        "fatalities_total": float(fat.sum(skipna=True)),
        "fatalities_mean_per_reported_incident": float(fat.mean()) if fat.notna().any() else None,
        "fatalities_median": float(fat.median()) if fat.notna().any() else None,
        "injuries_reported_n": int(inj.notna().sum()),
        "injuries_missing_n": int(inj.isna().sum()),
        "injuries_total": float(inj.sum(skipna=True)),
        "injuries_mean_per_reported_incident": float(inj.mean()) if inj.notna().any() else None,
        "injuries_median": float(inj.median()) if inj.notna().any() else None,
        "note": (
            "Summaries use only rows where the field was reported. Missing values "
            "are excluded rather than imputed as zero, so totals are lower bounds "
            "on the recorded figures in this dataset."
        ),
    }


def source_coverage(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("source_name", as_index=False)
        .agg(incidents=("incident_id", "count"))
        .sort_values("incidents", ascending=False)
    )
    return g.reset_index(drop=True)


def data_quality_checks(df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["incident_date"])
    future_cut = pd.Timestamp.now().normalize()
    return {
        "rows": int(len(df)),
        "duplicate_incident_ids": int(df["incident_id"].duplicated().sum()),
        "missing_date_n": int(df["incident_date"].isna().sum()),
        "future_dated_rows": int((dates > future_cut).sum()),
        "negative_fatalities": int((df["fatalities"] < 0).sum()),
        "negative_injuries": int((df["injuries"] < 0).sum()),
        "verification_status_counts": {
            str(k): int(v) for k, v in df["verification_status"].value_counts().items()
        },
    }


def frequency_change_stats(df: pd.DataFrame) -> dict:
    """Simple recorded-frequency change statistics across the full period."""
    yc = yearly_counts(df)
    if len(yc) < 2:
        return {"note": "Fewer than two years of data; change statistics not computed."}
    first_half = yc.iloc[: len(yc) // 2]["incidents"].mean()
    second_half = yc.iloc[len(yc) // 2 :]["incidents"].mean()
    peak = yc.loc[yc["incidents"].idxmax()]
    trough = yc.loc[yc["incidents"].idxmin()]
    return {
        "mean_yearly_incidents_first_half": float(round(first_half, 2)),
        "mean_yearly_incidents_second_half": float(round(second_half, 2)),
        "pct_change_first_to_second_half": (
            float(round((second_half - first_half) / max(first_half, 1e-9) * 100, 2))
        ),
        "peak_year": int(peak["year"]),
        "peak_year_incidents": int(peak["incidents"]),
        "lowest_year": int(trough["year"]),
        "lowest_year_incidents": int(trough["incidents"]),
        "interpretation_guardrail": (
            "Changes in RECORDED incident counts reflect both real-world change and "
            "changes in reporting coverage, source methodology, and classification. "
            "Do not treat this as a causal or predictive claim about any group."
        ),
    }


def run_full_eda(df: pd.DataFrame, report_path: Path | None = EDA_REPORT_PATH) -> dict:
    """Compute all EDA summaries and optionally persist them as JSON."""
    summary = {
        "overview": dataset_overview(df),
        "casualties": casualty_summary(df),
        "data_quality": data_quality_checks(df),
        "frequency_change": frequency_change_stats(df),
        "yearly_counts": yearly_counts(df).to_dict(orient="records"),
        "top_districts": district_summary(df).head(10).to_dict(orient="records"),
        "categories": category_distribution(df).to_dict(orient="records"),
        "sources": source_coverage(df).to_dict(orient="records"),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, pd.Timestamp):
                return o.isoformat()
            return str(o)

        report_path.write_text(json.dumps(summary, indent=2, default=_default), encoding="utf-8")
    return summary
