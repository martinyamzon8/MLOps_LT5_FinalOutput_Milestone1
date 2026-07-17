"""Basic unit tests for the PAL data pipeline (Milestone 1).

Run with: uv run pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dags"))

from pipeline_logic import extract_data, load_data, validate_data  # noqa: E402

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def extracted_df() -> pd.DataFrame:
    if not any(RAW_DATA_DIR.glob("*.xlsx")):
        pytest.skip("No sample data in data/raw/ -- add a sample xlsx to run this test.")
    return extract_data(RAW_DATA_DIR)


def test_extract_returns_nonempty_dataframe(extracted_df):
    assert len(extracted_df) > 0
    assert "PurchaseLeadTime" in extracted_df.columns
    assert "Group Status" in extracted_df.columns


def test_group_status_is_binary(extracted_df):
    assert set(extracted_df["Group Status"].unique()).issubset({0, 1})


def test_valid_data_passes_validation(extracted_df):
    validated = validate_data(extracted_df)
    assert len(validated) == len(extracted_df)


def test_invalid_cabin_fails_validation(extracted_df):
    bad = extracted_df.copy()
    bad.loc[bad.index[0], "Cabin"] = "NOT_A_REAL_CABIN"
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(bad)


def test_negative_fare_fails_validation(extracted_df):
    bad = extracted_df.copy()
    bad.loc[bad.index[0], "Average Fare"] = -10.0
    with pytest.raises(pandera.errors.SchemaError):
        validate_data(bad)


def test_load_writes_timestamped_parquet(tmp_path, extracted_df):
    validated = validate_data(extracted_df)
    out_path = load_data(validated, output_dir=tmp_path)
    assert out_path.exists()
    assert out_path.suffix == ".parquet"
    assert "pal_passengers_clean_" in out_path.name

    reloaded = pd.read_parquet(out_path)
    assert len(reloaded) == len(validated)
