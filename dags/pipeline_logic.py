"""
Core data pipeline logic for the PAL passenger segmentation project.

This module is intentionally kept free of Airflow imports. Airflow wires
these functions together as tasks in training_pipeline.py, but the logic
itself is plain pandas + Pandera so it can be unit tested quickly (see
tests/test_pipeline.py) without needing a full Airflow environment.

Pipeline stages:
    extract_data   -> read raw xlsx exports from data/raw
    validate_data  -> enforce a Pandera schema; raises on bad data
    load_data      -> write a timestamped, versioned parquet artifact
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# The raw exports are Tableau-style dumps: row 0 is a filter description,
# row 1 is blank, row 2 has the real column headers.
RAW_HEADER_SKIPROWS = 2

COLUMN_NAMES = [
    "Flight Date",
    "Route",
    "Entity",
    "Sub Entity",
    "Sector",
    "Flight Number",
    "Cabin",
    "Farebrand",
    "POS Region",
    "Ticketing Channel",
    "PNRCreationDate",
    "DOW",
    "POO",
    "PAX Count",
    "Average Fare",
]

VALID_CABINS = {"J", "W", "Y"}
VALID_DOW = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
def extract_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Read every xlsx export in raw_dir and concatenate into one dataframe.

    Also derives the two calculated fields called out in Deliverable 1:
    PurchaseLeadTime and Group Status.
    """
    files = sorted(raw_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No .xlsx files found in {raw_dir}. Did you place the raw "
            "exports in data/raw/?"
        )

    frames = []
    for f in files:
        df = pd.read_excel(f, skiprows=RAW_HEADER_SKIPROWS)
        df.columns = COLUMN_NAMES
        df = df.dropna(how="all")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Coerce types up front so validation checks numeric/date ranges
    # instead of string comparisons.
    combined["Flight Date"] = pd.to_datetime(combined["Flight Date"], errors="coerce")
    combined["PNRCreationDate"] = pd.to_datetime(
        combined["PNRCreationDate"], errors="coerce"
    )
    combined["PAX Count"] = pd.to_numeric(combined["PAX Count"], errors="coerce")
    combined["Average Fare"] = pd.to_numeric(combined["Average Fare"], errors="coerce")

    # Calculated fields from Deliverable 1
    combined["PurchaseLeadTime"] = (
        combined["Flight Date"] - combined["PNRCreationDate"]
    ).dt.days
    combined["Group Status"] = (combined["PAX Count"] > 1).astype(int)

    return combined


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
# Columns that are legitimately missing in the source data (confirmed by
# inspecting the real export) are marked nullable=True. Every other column
# is required. This is what makes the check "enforced": if data drifts
# outside these rules, .validate() raises SchemaError and the Airflow task
# fails instead of silently passing bad data downstream.
pal_schema = DataFrameSchema(
    {
        "Flight Date": Column(pa.DateTime, nullable=False),
        "Route": Column(str, Check.str_length(min_value=2), nullable=False),
        "Entity": Column(str, nullable=False),
        "Sub Entity": Column(str, nullable=False),
        "Sector": Column(str, nullable=False),
        "Flight Number": Column(str, nullable=False),
        "Cabin": Column(str, Check.isin(VALID_CABINS), nullable=False),
        "Farebrand": Column(str, nullable=False),
        "POS Region": Column(str, nullable=True),
        "Ticketing Channel": Column(str, nullable=True),
        "PNRCreationDate": Column(pa.DateTime, nullable=True),
        "DOW": Column(str, Check.isin(VALID_DOW), nullable=False),
        "POO": Column(str, nullable=True),
        "PAX Count": Column(int, Check.ge(1), nullable=False),
        "Average Fare": Column(float, Check.gt(0), nullable=False),
        "PurchaseLeadTime": Column(float, Check.ge(0), nullable=True),
        "Group Status": Column(int, Check.isin({0, 1}), nullable=False),
    },
    strict=False,
    coerce=False,
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate df against pal_schema. Raises pandera.errors.SchemaError
    on the first violation (lazy=False), which is what makes this an
    *enforced* check rather than a warning."""
    return pal_schema.validate(df, lazy=False)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data(df: pd.DataFrame, output_dir: Path = OUTPUT_DATA_DIR) -> Path:
    """Write df as a timestamped parquet file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"pal_passengers_clean_{run_id}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
