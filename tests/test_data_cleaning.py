"""Tests for the MAHSUD data-cleaning and data-loading pipeline."""
import numpy as np
import pandas as pd
import pytest

from config import REQUIRED_COLUMNS
from src.data_cleaning import (
    clean_incident_data,
    detect_and_drop_duplicates,
    normalize_districts,
    parse_and_normalize_dates,
)
from src.data_collection import generate_demo_dataset, validate_schema
from src.exploratory_analysis import (
    casualty_summary,
    dataset_overview,
    district_summary,
    filter_incidents,
    yearly_counts,
)
from src.feature_engineering import (
    build_feature_matrix,
    build_monthly_series,
    chronological_split,
)


def _minimal_rows():
    return pd.DataFrame(
        [
            {
                "incident_id": "T-1",
                "incident_date": "2015-03-05",
                "district": "peshawar",
                "province": "Khyber Pakhtunkhwa",
                "incident_category": "armed_attack",
                "fatalities": 2,
                "injuries": 4,
                "source_name": "unit-test",
                "source_url": "",
                "verification_status": "unverified",
            },
            {
                "incident_id": "T-2",
                "incident_date": "2015-03-05",
                "district": "Peshawar",
                "province": "Khyber Pakhtunkhwa",
                "incident_category": "armed_attack",
                "fatalities": 2,
                "injuries": 4,
                "source_name": "unit-test",
                "source_url": "",
                "verification_status": "unverified",
            },
            {
                "incident_id": "T-3",
                "incident_date": "not-a-date",
                "district": "Swat",
                "province": "Khyber Pakhtunkhwa",
                "incident_category": "unknown",
                "fatalities": np.nan,
                "injuries": np.nan,
                "source_name": "unit-test",
                "source_url": "",
                "verification_status": "unverified",
            },
        ]
    )


def test_generate_demo_dataset_is_labeled_and_well_formed():
    df = generate_demo_dataset(n_duplicates=5, n_missing_dates=2, casualty_missing_rate=0.1, seed=0)
    assert len(df) > 100
    raw_required = [c for c in REQUIRED_COLUMNS if c not in ("year", "month")]
    assert set(raw_required).issubset(df.columns)
    assert (df["verification_status"] == "demonstration_synthetic").all()
    assert (df["province"] == "Khyber Pakhtunkhwa").all()


def test_validate_schema_flags_missing_columns():
    df = _minimal_rows().drop(columns=["fatalities"])
    problems = validate_schema(df)
    assert any("Missing required columns" in p for p in problems)


def test_date_parsing_drops_unparseable_and_adds_year_month():
    df, report = parse_and_normalize_dates(_minimal_rows())
    assert report["dropped_unparseable_dates"] == 1
    assert len(df) == 2
    assert (df["year"] == 2015).all()
    assert (df["month"] == 3).all()
    assert pd.api.types.is_datetime64_any_dtype(df["incident_date"])


def test_district_normalization_case_insensitive():
    df, report = normalize_districts(_minimal_rows())
    assert (df["district"] == "Peshawar").sum() == 2
    assert "Swat" in report["distinct_districts_after"]


def test_duplicate_detection_on_content_keys():
    df, _ = normalize_districts(_minimal_rows())
    df, report = detect_and_drop_duplicates(df)
    assert report["duplicate_content_rows"] == 1
    assert len(df) == 2


def test_missing_casualties_stay_nan_not_zero():
    cleaned, _ = clean_incident_data(_minimal_rows(), report_path=None)
    t3 = cleaned[cleaned["incident_id"] == "T-3"]
    if len(t3):
        assert pd.isna(t3["fatalities"].iloc[0])
    assert (cleaned["fatalities"].fillna(-1) >= 0).all()


def test_yearly_counts_and_casualty_summary():
    cleaned, _ = clean_incident_data(_minimal_rows(), report_path=None)
    yearly = yearly_counts(cleaned)
    assert yearly["incidents"].sum() == len(cleaned)
    summary = casualty_summary(cleaned)
    assert summary["fatalities_total"] >= 2
    assert summary["fatalities_missing_n"] >= 0


def test_filter_incidents_by_district_and_date():
    cleaned, _ = clean_incident_data(_minimal_rows(), report_path=None)
    only_peshawar = filter_incidents(cleaned, districts=["Peshawar"])
    assert set(only_peshawar["district"]) == {"Peshawar"}
    windowed = filter_incidents(cleaned, date_start="2015-01-01", date_end="2015-12-31")
    assert (windowed["year"] == 2015).all()


def test_monthly_series_is_regular_and_aggregates():
    cleaned, _ = clean_incident_data(_minimal_rows(), report_path=None)
    series = build_monthly_series(cleaned)
    assert series["incident_count"].sum() == len(cleaned)
    expected = pd.period_range(series["period"].min(), series["period"].max(), freq="M")
    assert [str(p) for p in series["period"]] == [str(p) for p in expected]


def test_chronological_split_no_overlap_and_order():
    df = generate_demo_dataset(n_duplicates=0, n_missing_dates=0, casualty_missing_rate=0.0, seed=1)
    cleaned, _ = clean_incident_data(df, report_path=None)
    series = build_monthly_series(cleaned)
    feats, cols = build_feature_matrix(series)
    train, test = chronological_split(feats, test_months=12)
    assert len(test) == 12
    assert train["month_start"].max() < test["month_start"].min()
    assert set(cols).issubset(feats.columns)
    assert feats[cols].isna().sum().sum() == 0


def test_empty_dataset_raises_helpfully():
    empty = _minimal_rows().iloc[0:0]
    with pytest.raises(ValueError):
        clean_incident_data(empty, report_path=None)
    overview = dataset_overview(empty)
    assert overview["total_recorded_incidents"] == 0
