"""
Integration test for the PAL passenger data pipeline (Deliverable 2c).

Runs extract -> validate -> load end-to-end against a synthetic Excel file
generated inside the test, so it requires no real data, no Airflow, and no
MLflow server. This is the only test in the suite that runs unconditionally
in CI.

Run with: uv run pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dags"))

# isort: off
from pipeline_logic import extract_data, load_data, validate_data  # noqa: E402
# isort: on


# ---------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------
# extract_data() calls pd.read_excel(..., skiprows=2) because the real
# Tableau exports carry a filter description in row 0 and a blank row 1.
# The synthetic file must reproduce that shape or the header row is eaten.
SYNTHETIC_ROW_COUNT = 4


def _write_synthetic_export(raw_dir: Path) -> Path:
    """Write one Tableau-shaped .xlsx into raw_dir and return its path."""
    rows = [
        ["2026-01-05", "MNL-CEB", "PAL", "Domestic", "Luzon-Visayas", "PR2841",
         "Y", "Promo", "PH", "Web", "2025-12-20", "Mon", "MNL", 2, 3512.75],
        ["2026-01-06", "MNL-DVO", "PAL", "Domestic", "Luzon-Mindanao", "PR2812",
         "J", "Flex", "PH", "GDS", "2025-12-01", "Tue", "MNL", 1, 12847.50],
        ["2026-01-07", "MNL-ILO", "PAL", "Domestic", "Luzon-Visayas", "PR2337",
         "W", "Premium", "SG", "Agency", "2025-11-15", "Wed", "MNL", 4, 7183.25],
        # PNRCreationDate deliberately blank: forces PurchaseLeadTime to float,
        # which is what the schema declares. See the note in the test below.
        ["2026-01-08", "MNL-BCD", "PAL", "Domestic", "Luzon-Visayas", "PR2141",
         "Y", "Promo", "PH", "Web", None, "Thu", "MNL", 1, 2946.80],
    ]

    junk = ["Filter: Domestic / Manila Hub / CY2026"] + [None] * 14
    blank = [None] * 15
    header = [
        "Flight Date", "Route", "Entity", "Sub Entity", "Sector",
        "Flight Number", "Cabin", "Farebrand", "POS Region",
        "Ticketing Channel", "PNRCreationDate", "DOW", "POO",
        "PAX Count", "Average Fare",
    ]

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "synthetic_export.xlsx"
    pd.DataFrame([junk, blank, header, *rows]).to_excel(
        out_path, index=False, header=False
    )
    return out_path


# ---------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------
def test_pipeline_end_to_end_produces_artifact(tmp_path: Path) -> None:
    """extract -> validate -> load runs clean and writes a readable parquet.

    Chains the three pipeline_logic functions in the same order the Airflow
    DAG runs them. It does not exercise the DAG itself: training_pipeline.py
    imports airflow, which is not available on a CI runner, and the tasks
    add only parquet round-trips between stages.
    """
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _write_synthetic_export(raw_dir)

    # 1. Extract
    extracted = extract_data(raw_dir=raw_dir)
    assert len(extracted) == SYNTHETIC_ROW_COUNT
    assert "PurchaseLeadTime" in extracted.columns
    assert "Group Status" in extracted.columns

    # 2. Validate — raises SchemaError if the contract is broken
    validated = validate_data(extracted)
    assert len(validated) == SYNTHETIC_ROW_COUNT

    # 3. Load
    out_path = load_data(validated, output_dir=output_dir)

    # The artifact exists, is named as expected, and survives a round-trip
    assert out_path.exists()
    assert out_path.suffix == ".parquet"
    assert "pal_passengers_clean_" in out_path.name

    reloaded = pd.read_parquet(out_path)
    assert len(reloaded) == SYNTHETIC_ROW_COUNT
    assert list(reloaded.columns) == list(validated.columns)