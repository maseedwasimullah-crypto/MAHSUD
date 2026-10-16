"""Data acquisition, source documentation, and demonstration dataset generation.

IMPORTANT SAFETY AND INTEGRITY NOTES
-----------------------------------
* MAHSUD is a historical, aggregate research and public-awareness tool.
* This module NEVER presents synthetic records as real incidents. Every
  demonstration row carries verification_status="demonstration_synthetic".
* Replace the demonstration CSV with verified data before drawing any
  real-world conclusion. See data/sample/DATASET_PROVENANCE.md.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEMO_DATA_PATH, PROVINCE_NAME, RANDOM_STATE, REQUIRED_COLUMNS, RAW_DATA_PATH

DEMO_STATUS = "demonstration_synthetic"
DEMO_SOURCE_NAME = "MAHSUD synthetic demonstration dataset (NOT real incidents)"
DEMO_SOURCE_URL = "file:data/sample/DATASET_PROVENANCE.md"

PUBLIC_SOURCES = [
    {
        "source_name": "Pakistan Institute for Peace Studies (PIPS) - Pakistan Security Report (annual)",
        "url": "https://www.ips.org.pk/",
        "coverage": "National and provincial incident aggregates, typically one calendar year per report",
        "retrieval_note": "Published as annual PDF reports; requires manual extraction into CSV",
        "limitations": (
            "Aggregate tables rather than incident-level rows; definitions of incident "
            "types may differ from other sources; occasional revisions between editions."
        ),
        "license_terms": "Check publisher terms before redistribution of extracted tables.",
    },
    {
        "source_name": "South Asia Terrorism Portal (SATP) - Incident data for KPK",
        "url": "https://www.satp.org/",
        "coverage": "District-wise incident listings for Pakistan provinces, updated periodically",
        "retrieval_note": "Website data extraction must comply with SATP terms of use",
        "limitations": (
            "Third-party compilation; coverage completeness varies by year and district; "
            "terms of use may restrict bulk scraping or redistribution."
        ),
        "license_terms": "Review https://www.satp.org/disclaimers before any use.",
    },
    {
        "source_name": "National Counter Terrorism Authority (NACTA), Government of Pakistan",
        "url": "https://nacta.gov.pk/",
        "coverage": "Official publications and press data on terrorism in Pakistan",
        "retrieval_note": "Official publications must be downloaded manually when available",
        "limitations": "Publication cadence and granularity vary; incident-level open data is limited.",
        "license_terms": "Government publications; verify reuse terms per document.",
    },
    {
        "source_name": "START Global Terrorism Database (GTD)",
        "url": "https://www.start.umd.edu/gtd/",
        "coverage": "Global incident-level terrorism data 1970 onward, including Pakistan",
        "retrieval_note": "Registration and data-use agreement required for download",
        "limitations": (
            "Open-access but requires agreement; coding decisions by GTD team; "
            "recent years may lag; filtering to KPK requires province mapping."
        ),
        "license_terms": "GTD data-use terms apply.",
    },
    {
        "source_name": "UNODC Global Study on Homicide / terrorism statistical annexes",
        "url": "https://www.unodc.org/unodc/en/data-and-analysis/statistics.html",
        "coverage": "National-level terrorism fatality statistics for Pakistan",
        "retrieval_note": "Static statistical releases",
        "limitations": "National level only; not district-level; lag of typically one or more years.",
        "license_terms": "UNODC open data terms.",
    },
]

DEMO_YEARLY_INCIDENT_RATE = {
    2002: 2, 2003: 3, 2004: 4, 2005: 5, 2006: 7, 2007: 12, 2008: 25,
    2009: 45, 2010: 55, 2011: 60, 2012: 70, 2013: 80, 2014: 65,
    2015: 35, 2016: 28, 2017: 25, 2018: 30, 2019: 28, 2020: 18,
    2021: 20, 2022: 28, 2023: 40, 2024: 45,
}

DEMO_DISTRICT_WEIGHTS = {
    "Peshawar": 0.11, "Swat": 0.08, "Bannu": 0.07, "Dera Ismail Khan": 0.06,
    "Kohat": 0.06, "Mardan": 0.05, "Abbottabad": 0.04, "Lower Dir": 0.04,
    "Buner": 0.04, "Tank": 0.035, "Lakki Marwat": 0.035, "Hangu": 0.035,
    "Kurram": 0.035, "North Waziristan": 0.05, "South Waziristan": 0.04,
    "Khyber": 0.035, "Mohmand": 0.025, "Bajaur": 0.025, "Chitral": 0.02,
    "Shangla": 0.025, "Swabi": 0.025, "Nowshera": 0.025, "Haripur": 0.02,
    "Battagram": 0.015, "Upper Dir": 0.025, "Karak": 0.025, "Torghar": 0.01,
    "Mansehra": 0.02, "Charsadda": 0.025,
}

DEMO_CATEGORY_WEIGHTS = {
    "armed_attack": 0.28,
    "explosive_device": 0.26,
    "security_forces_clash": 0.18,
    "sectarian_violence": 0.08,
    "suicide_attack": 0.03,
    "civil_unrest": 0.07,
    "unknown": 0.10,
}

MONTH_SEASONALITY = np.array(
    [0.9, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9]
)


def get_source_registry() -> list[dict]:
    """Return the documented list of candidate public data sources."""
    return [dict(s) for s in PUBLIC_SOURCES]


def generate_demo_dataset(
    n_duplicates: int = 30,
    n_missing_dates: int = 5,
    casualty_missing_rate: float = 0.08,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Build a clearly labeled synthetic incident-level dataset for KPK.

    The series is statistically plausible (long-run rise to ~2013-2014 and
    decline thereafter, matching the *shape* of published national and
    provincial trend curves) but contains NO real incidents. It exists so the
    analysis, modeling, and dashboard pipeline can be developed and tested
    offline until verified data is supplied.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    districts = list(DEMO_DISTRICT_WEIGHTS.keys())
    d_weights = np.array([DEMO_DISTRICT_WEIGHTS[d] for d in districts], dtype=float)
    d_weights = d_weights / d_weights.sum()
    categories = list(DEMO_CATEGORY_WEIGHTS.keys())
    c_weights = np.array([DEMO_CATEGORY_WEIGHTS[c] for c in categories], dtype=float)
    c_weights = c_weights / c_weights.sum()

    incident_number = 1
    for year, monthly_rate in DEMO_YEARLY_INCIDENT_RATE.items():
        for month in range(1, 13):
            lam = monthly_rate * MONTH_SEASONALITY[month - 1]
            n = int(rng.poisson(lam))
            if n == 0:
                continue
            days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
            for _ in range(n):
                district = str(rng.choice(districts, p=d_weights))
                category = str(rng.choice(categories, p=c_weights))
                day = int(rng.integers(1, days_in_month + 1))
                if category == "suicide_attack":
                    fatalities = int(rng.poisson(4.5))
                    injuries = int(rng.poisson(8.0))
                elif category == "explosive_device":
                    fatalities = int(rng.poisson(2.0))
                    injuries = int(rng.poisson(3.5))
                elif category == "armed_attack":
                    fatalities = int(rng.poisson(1.6))
                    injuries = int(rng.poisson(2.0))
                elif category == "security_forces_clash":
                    fatalities = int(rng.poisson(1.2))
                    injuries = int(rng.poisson(1.5))
                elif category == "sectarian_violence":
                    fatalities = int(rng.poisson(2.5))
                    injuries = int(rng.poisson(2.5))
                else:
                    fatalities = int(rng.poisson(0.6))
                    injuries = int(rng.poisson(0.9))
                if rng.random() < 0.15:
                    fatalities = 0
                if rng.random() < 0.2:
                    injuries = 0
                rows.append(
                    {
                        "incident_id": f"DEMO-{incident_number:06d}",
                        "incident_date": f"{year}-{month:02d}-{day:02d}",
                        "district": district,
                        "province": PROVINCE_NAME,
                        "incident_category": category,
                        "fatalities": fatalities,
                        "injuries": injuries,
                        "source_name": DEMO_SOURCE_NAME,
                        "source_url": DEMO_SOURCE_URL,
                        "verification_status": DEMO_STATUS,
                    }
                )
                incident_number += 1

    df = pd.DataFrame(rows)
    n_casualty_missing = int(len(df) * casualty_missing_rate)
    casualty_idx = rng.choice(len(df), size=n_casualty_missing, replace=False)
    col = rng.choice(["fatalities", "injuries"], size=n_casualty_missing)
    for i, c in zip(casualty_idx, col):
        df.loc[i, c] = np.nan

    if n_duplicates > 0 and len(df) > 0:
        dup_source = df.sample(n=min(n_duplicates, len(df)), random_state=seed)
        dup_rows = dup_source.copy()
        dup_rows["incident_id"] = [f"DEMO-DUP-{i:04d}" for i in range(len(dup_rows))]
        df = pd.concat([df, dup_rows], ignore_index=True)

    if n_missing_dates > 0 and len(df) > n_missing_dates:
        miss_idx = rng.choice(len(df), size=n_missing_dates, replace=False)
        df.loc[miss_idx, "incident_date"] = ""

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["incident_id"] = [f"DEMO-{i:06d}" for i in range(1, len(df) + 1)]
    return df


def write_demo_dataset(
    out_path: Path = DEMO_DATA_PATH,
    n_duplicates: int = 30,
    n_missing_dates: int = 5,
    casualty_missing_rate: float = 0.08,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate the demonstration dataset, write CSV + provenance file."""
    df = generate_demo_dataset(
        n_duplicates=n_duplicates,
        n_missing_dates=n_missing_dates,
        casualty_missing_rate=casualty_missing_rate,
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    provenance = {
        "dataset_name": "MAHSUD demonstration dataset (SYNTHETIC)",
        "is_real_incident_data": False,
        "warning": (
            "Every record in this file is computer-generated for software testing. "
            "None of these rows correspond to real-world events, people, or locations "
            "experiencing harm. Do not cite these numbers in research or journalism."
        ),
        "generated_on": date.today().isoformat(),
        "generator": "src/data_collection.py::generate_demo_dataset",
        "random_seed": seed,
        "date_range": f"{df['incident_date'].replace('', np.nan).min()} to {df['incident_date'].replace('', np.nan).max()}",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "intentional_flaws": [
            f"{n_duplicates} duplicate rows (new IDs, identical content)",
            f"{n_missing_dates} rows with missing incident_date",
            f"{casualty_missing_rate:.0%} missing values in fatalities/injuries "
            "(missing must NOT be treated as zero)",
        ],
        "how_to_replace_with_real_data": [
            "1. Obtain verified data from a documented source (see PUBLIC_SOURCES in src/data_collection.py).",
            "2. Export it as CSV using the same column names as config.REQUIRED_COLUMNS.",
            "3. Save it as data/raw/kpk_incidents.csv (takes priority over the demo file).",
            "4. Set verification_status on each row honestly (e.g., 'verified', 'unverified', 'media_report').",
            "5. Update the provenance table in data/sample/DATASET_PROVENANCE.md and the Streamlit Sources page.",
        ],
    }
    prov_path = out_path.parent / "DATASET_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return df


DERIVED_COLUMNS = {"year", "month"}


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of schema problems (empty list means valid).

    year/month are treated as derived: the cleaning pipeline recomputes them
    from incident_date, so raw files need not contain them.
    """
    problems: list[str] = []
    required_now = [c for c in REQUIRED_COLUMNS if c not in DERIVED_COLUMNS]
    missing = [c for c in required_now if c not in df.columns]
    if missing:
        problems.append(f"Missing required columns: {missing}")
    if "incident_date" not in df.columns:
        problems.append("Missing required column: incident_date (needed to derive year/month)")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        problems.append(f"Unexpected extra columns (kept but ignored downstream): {extra}")
    if len(df) == 0:
        problems.append("Dataset has zero rows.")
    return problems


def load_incident_data(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load incident data, preferring real raw data, else the demo dataset.

    Returns the dataframe and a provenance dict describing which file was used.
    """
    chosen: Path | None = None
    if path is not None:
        chosen = Path(path)
    elif RAW_DATA_PATH.exists():
        chosen = RAW_DATA_PATH
    elif DEMO_DATA_PATH.exists():
        chosen = DEMO_DATA_PATH
    if chosen is None or not chosen.exists():
        raise FileNotFoundError(
            "No dataset found. Run: python -m src.data_collection (creates the "
            "demonstration dataset), or place verified data at data/raw/kpk_incidents.csv."
        )

    df = pd.read_csv(chosen, dtype={"district": "string", "incident_category": "string"})
    problems = validate_schema(df)
    hard = [p for p in problems if p.startswith("Missing") or p.startswith("Dataset")]
    if hard:
        raise ValueError(f"Schema validation failed: {hard}")

    is_demo = chosen == DEMO_DATA_PATH
    provenance = {
        "file_used": str(chosen),
        "data_type": "synthetic_demonstration" if is_demo else "user_supplied_verify_manually",
        "is_real_incident_data": not is_demo,
        "rows": int(len(df)),
        "date_min": str(df["incident_date"].replace("", np.nan).dropna().min()) if "incident_date" in df else None,
        "date_max": str(df["incident_date"].replace("", np.nan).dropna().max()) if "incident_date" in df else None,
        "schema_warnings": [p for p in problems if p not in hard],
    }
    return df, provenance


def _main() -> None:
    df = write_demo_dataset()
    print(f"Wrote demonstration dataset: {DEMO_DATA_PATH} ({len(df)} rows)")
    print("WARNING: this dataset is SYNTHETIC and must never be cited as real incidents.")


if __name__ == "__main__":
    _main()
