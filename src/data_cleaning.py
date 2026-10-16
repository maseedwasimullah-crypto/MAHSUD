"""Data cleaning pipeline for MAHSUD incident records.

Cleaning decisions are recorded and exported as a JSON report so the
process is auditable. Missing casualty values are preserved as missing
(NaN) and are never silently converted to zero.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLEANING_REPORT_PATH, PROCESSED_DATA_PATH, REQUIRED_COLUMNS

DERIVED_COLUMNS = {"year", "month"}

DISTRICT_FIXES = {
    "d.i. khan": "Dera Ismail Khan",
    "di khan": "Dera Ismail Khan",
    "dera ismail khan": "Dera Ismail Khan",
    "d. i. khan": "Dera Ismail Khan",
    "peshawar": "Peshawar",
    "north waziristan": "North Waziristan",
    "south waziristan": "South Waziristan",
    "lower dir": "Lower Dir",
    "upper dir": "Upper Dir",
    "lakki marwat": "Lakki Marwat",
    "nowshera": "Nowshera",
    "charsadda": "Charsadda",
}

DEDUP_KEYS = ["incident_date", "district", "incident_category", "fatalities", "injuries"]


def parse_and_normalize_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Parse incident_date; rows without a parseable date are dropped."""
    report: dict = {"rows_in": int(len(df))}
    out = df.copy()
    parsed = pd.to_datetime(out["incident_date"], errors="coerce")
    n_bad = int(parsed.isna().sum())
    report["dropped_unparseable_dates"] = n_bad
    out = out.loc[parsed.notna()].copy()
    out["incident_date"] = parsed.loc[parsed.notna()]
    out["year"] = out["incident_date"].dt.year.astype(int)
    out["month"] = out["incident_date"].dt.month.astype(int)
    report["rows_out"] = int(len(out))
    return out, report


def normalize_districts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize district spelling/casing via lookup then title-case fallback."""
    out = df.copy()
    raw = out["district"].astype("string").str.strip()
    lowered = raw.str.lower()
    mapped = lowered.map(DISTRICT_FIXES)
    out["district"] = mapped.fillna(raw.str.title()).astype("string")
    n_unknown = int(out["district"].isna().sum() + (out["district"].str.len() == 0).sum())
    report = {
        "distinct_districts_after": sorted(out["district"].dropna().unique().tolist()),
        "rows_with_missing_district": n_unknown,
    }
    return out, report


def coerce_numeric_casualties(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Coerce fatalities/injuries to numeric, keeping missing as NaN."""
    out = df.copy()
    report: dict = {}
    for col in ("fatalities", "injuries"):
        before_null = int(out[col].isna().sum())
        coerced = pd.to_numeric(out[col], errors="coerce")
        became_null = int(coerced.isna().sum() - before_null)
        out[col] = coerced.astype("float64")
        report[col] = {
            "originally_missing": before_null,
            "non_numeric_coerced_to_missing": became_null,
            "total_missing_after": int(out[col].isna().sum()),
        }
    return out, report


def detect_and_drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact-ID duplicates, then content duplicates on DEDUP_KEYS."""
    out = df.copy()
    report: dict = {}
    n_before = len(out)
    report["duplicate_incident_id_rows"] = int(out["incident_id"].duplicated().sum())
    out = out.drop_duplicates(subset=["incident_id"], keep="first")
    content_mask = out.duplicated(subset=DEDUP_KEYS, keep="first")
    report["duplicate_content_rows"] = int(content_mask.sum())
    out = out.loc[~content_mask].copy()
    report["rows_removed_total"] = n_before - len(out)
    report["rows_remaining"] = int(len(out))
    return out.reset_index(drop=True), report


def drop_missing_district_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    mask = out["district"].isna() | (out["district"].astype("string").str.strip() == "")
    n = int(mask.sum())
    out = out.loc[~mask].copy()
    return out, {"dropped_missing_district_rows": n}


def clean_incident_data(
    df: pd.DataFrame, report_path: Path | None = CLEANING_REPORT_PATH
) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline and persist an auditable report."""
    steps: dict = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_received": int(len(df)),
        "columns_received": list(df.columns),
        "note": (
            "Missing fatalities/injuries are kept as NaN on purpose: absence of a "
            "reported figure is not evidence of zero casualties."
        ),
    }
    working = df.copy()
    if len(working) == 0:
        raise ValueError("Cannot clean an empty dataset (zero rows).")
    required_now = [c for c in REQUIRED_COLUMNS if c not in DERIVED_COLUMNS]
    missing_required = [c for c in required_now if c not in working.columns]
    if missing_required:
        raise ValueError(f"Cannot clean data, missing columns: {missing_required}")

    working, s = coerce_numeric_casualties(working)
    steps["numeric_coercion"] = s

    working, s = parse_and_normalize_dates(working)
    steps["date_parsing"] = s

    working, s = normalize_districts(working)
    steps["district_normalization"] = s

    working, s = detect_and_drop_duplicates(working)
    steps["deduplication"] = s

    working, s = drop_missing_district_rows(working)
    steps["missing_districts"] = s

    working = working.sort_values("incident_date").reset_index(drop=True)
    steps["rows_after_cleaning"] = int(len(working))
    steps["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(steps, indent=2), encoding="utf-8")
    return working, steps


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> Path:
    out = df.copy()
    out["incident_date"] = pd.to_datetime(out["incident_date"]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def load_processed(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"district": "string", "incident_category": "string"})
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    return df
