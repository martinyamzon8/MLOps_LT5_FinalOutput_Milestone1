"""Data validation tests for PAL passenger data schema (Milestone 1 & 2 Deliverable).

Checks Pandera schema enforcement on valid and invalid records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dags"))

from pipeline_logic import validate_data  # noqa: E402


@pytest.fixture(scope="module")
def sample_valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid DataFrame conforming to the schema."""
    return pd.DataFrame(
        {
            "Flight Date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "Route": ["MNL-CEB", "MNL-DVO"],
            "Entity": ["PAL", "PAL"],
            "Sub Entity": ["Domestic", "Domestic"],
            "Sector": ["Luzon-Visayas", "Luzon-Mindanao"],
            "Flight Number": ["PR2841", "PR2812"],
            "Cabin": ["Y", "J"],
            "Farebrand": ["Promo", "Flex"],
            "POS Region": ["PH", "PH"],
            "Ticketing Channel": ["Web", "GDS"],
            "PNRCreationDate": pd.to_datetime(["2026-02-15", "2026-02-01"]),
            "DOW": ["Sun", "Mon"],
            "POO": ["MNL", "MNL"],
            "PAX Count": [2, 1],
            "Average Fare": [3500.0, 12000.0],
            "PurchaseLeadTime": [14.0, 31.0],
            "Group Status": [1, 0],
        }
    )


def test_valid_data_passes_validation(sample_valid_dataframe: pd.DataFrame):
    """Deliverable requirement: valid data passes through schema check."""
    validated = validate_data(sample_valid_dataframe)
    assert len(validated) == len(sample_valid_dataframe)


def test_invalid_cabin_fails_validation(sample_valid_dataframe: pd.DataFrame):
    """Deliverable requirement: schema rejects invalid categorical cabin codes."""
    bad = sample_valid_dataframe.copy()
    bad.loc[bad.index[0], "Cabin"] = "INVALID_CABIN"
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(bad)


def test_negative_fare_fails_validation(sample_valid_dataframe: pd.DataFrame):
    """Deliverable requirement: schema rejects negative fare values."""
    bad = sample_valid_dataframe.copy()
    bad.loc[bad.index[0], "Average Fare"] = -50.0
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(bad)


def test_zero_pax_count_fails_validation(sample_valid_dataframe: pd.DataFrame):
    """Deliverable requirement: schema rejects PAX count < 1."""
    bad = sample_valid_dataframe.copy()
    bad.loc[bad.index[0], "PAX Count"] = 0
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(bad)


def test_invalid_dow_fails_validation(sample_valid_dataframe: pd.DataFrame):
    """Deliverable requirement: schema rejects invalid Day of Week."""
    bad = sample_valid_dataframe.copy()
    bad.loc[bad.index[0], "DOW"] = "Funday"
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(bad)
