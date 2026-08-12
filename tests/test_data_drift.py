"""Data drift monitoring tests using Evidently AI.

This test module:
1. Uses the PAL passenger data in the data folder as the reference benchmark.
2. Evaluates incoming/current data against the benchmark.
3. Flags any features that exceed the 15% drift threshold (drift_share=0.15).
4. Generates a human-readable Evidently HTML report for interactive visual review.

Run with:
    uv run pytest tests/test_data_drift.py -v
or directly:
    uv run python tests/test_data_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report
from evidently.legacy.test_preset import DataDriftTestPreset
from evidently.legacy.test_suite import TestSuite as EvTestSuite
from evidently.legacy.tests import TestShareOfDriftedColumns as EvTestShareOfDriftedColumns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dags"))

from pipeline_logic import RAW_DATA_DIR, extract_data, validate_data  # noqa: E402

# Tell pytest not to treat imported Evidently classes as pytest test classes
EvTestSuite.__test__ = False  # type: ignore[attr-defined]
EvTestShareOfDriftedColumns.__test__ = False  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_PARQUET = DATA_DIR / "benchmark_data.parquet"
PROCESSED_PARQUET = DATA_DIR / "processed" / "pal_passengers_clean_latest.parquet"
SAMPLE_DRIFTED_PARQUET = DATA_DIR / "sample_drifted_data.parquet"
SAMPLE_DRIFTED_CSV = DATA_DIR / "sample_drifted_data.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_HTML_PATH = REPORTS_DIR / "data_drift_report.html"
TEST_SUITE_HTML_PATH = REPORTS_DIR / "data_drift_test_suite.html"

# Default features to monitor across the passenger dataset
MONITORED_FEATURES = [
    "Average Fare",
    "PurchaseLeadTime",
    "PAX Count",
    "Cabin",
    "Farebrand",
    "DOW",
    "POO",
    "POS Region",
    "Ticketing Channel",
    "Group Status",
]

DRIFT_SHARE_THRESHOLD = 0.15  # Flag if >= 15% of features drift
BENCHMARK_SAMPLE_SIZE = 10000
SAMPLE_DATASET_SIZE = 5000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data Loaders & Synthetic Drift Generator
# ---------------------------------------------------------------------------
def load_benchmark_data(
    sample_size: int | None = BENCHMARK_SAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Load benchmark reference data from parquet or raw files."""
    if BENCHMARK_PARQUET.exists():
        df = pd.read_parquet(BENCHMARK_PARQUET)
    elif PROCESSED_PARQUET.exists():
        df = pd.read_parquet(PROCESSED_PARQUET)
    elif any(RAW_DATA_DIR.glob("*.xlsx")):
        df = validate_data(extract_data(RAW_DATA_DIR))
    else:
        raise FileNotFoundError(
            "No benchmark data found in data/benchmark_data.parquet, data/processed/, "
            "or data/raw/."
        )

    if sample_size and len(df) > sample_size:
        return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df.reset_index(drop=True)


def generate_sample_drifted_dataset(
    benchmark_df: pd.DataFrame,
    sample_size: int = SAMPLE_DATASET_SIZE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a sample current/production dataset based on the benchmark,

    with deliberate, controlled drift exceeding 15% in specific features:
    - 'Average Fare': +30% price inflation (shifts numerical distribution)
    - 'PurchaseLeadTime': +15 days increase in booking lead time
    - 'Cabin': 25% shift from Economy (Y) to Business (J) (shifts categorical distribution)
    - Other features ('PAX Count', 'Group Status', 'DOW', etc.) remain stable.
    """
    n = min(sample_size, len(benchmark_df))
    sample = benchmark_df.sample(n=n, random_state=random_state).copy().reset_index(drop=True)

    # 1. Numeric drift: Average Fare +30%
    if "Average Fare" in sample.columns:
        sample["Average Fare"] = (sample["Average Fare"] * 1.30).round(2)

    # 2. Numeric drift: PurchaseLeadTime +15 days
    if "PurchaseLeadTime" in sample.columns:
        sample["PurchaseLeadTime"] = sample["PurchaseLeadTime"].apply(
            lambda x: max(0.0, float(x) + 15.0) if pd.notna(x) else x
        )

    # 3. Categorical drift: Shift 25% of Economy (Y) bookings to Business (J)
    if "Cabin" in sample.columns:
        mask_y = sample["Cabin"] == "Y"
        y_indices = sample[mask_y].index
        drift_count = int(len(y_indices) * 0.25)
        if drift_count > 0:
            change_indices = y_indices[:drift_count]
            sample.loc[change_indices, "Cabin"] = "J"

    return sample


def get_or_create_sample_dataset() -> pd.DataFrame:
    """Load existing sample drifted dataset or generate and save a new one."""
    if SAMPLE_DRIFTED_PARQUET.exists():
        return pd.read_parquet(SAMPLE_DRIFTED_PARQUET)

    benchmark = load_benchmark_data(sample_size=None)
    sample = generate_sample_drifted_dataset(benchmark)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_DRIFTED_PARQUET, index=False)
    sample.to_csv(SAMPLE_DRIFTED_CSV, index=False)
    return sample


# ---------------------------------------------------------------------------
# Core Evidently Drift Monitoring Logic
# ---------------------------------------------------------------------------
def run_drift_monitoring(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
    save_html: bool = True,
    report_path: Path = REPORT_HTML_PATH,
    test_suite_path: Path = TEST_SUITE_HTML_PATH,
) -> dict[str, Any]:
    """Run Evidently data drift analysis comparing current_df against reference_df.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Benchmark reference dataset.
    current_df : pd.DataFrame
        Current / incoming production sample dataset.
    features : list[str], optional
        List of feature columns to monitor. Defaults to MONITORED_FEATURES.
    drift_share_threshold : float
        Proportion threshold (e.g. 0.15 = 15%) for flagging dataset-level drift.
    save_html : bool
        Whether to save interactive HTML reports for human review.
    report_path : Path
        Destination path for the Evidently Data Drift Report HTML.
    test_suite_path : Path
        Destination path for the Evidently Test Suite HTML.

    Returns
    -------
    dict[str, Any]
        Structured drift results including dataset drift status, flagged features,
        and per-column drift scores and stattest names.
    """
    if features is None:
        features = [
            col
            for col in MONITORED_FEATURES
            if col in reference_df.columns and col in current_df.columns
        ]

    ref_subset = reference_df[features].copy()
    curr_subset = current_df[features].copy()

    # 1. Generate Evidently Data Drift Report (visual distributions & stattests)
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    report.run(reference_data=ref_subset, current_data=curr_subset)

    # 2. Generate Evidently Test Suite (executable pass/fail assertions)
    suite = EvTestSuite(
        tests=[
            DataDriftTestPreset(drift_share=drift_share_threshold),
            EvTestShareOfDriftedColumns(lt=drift_share_threshold),
        ]
    )
    suite.run(reference_data=ref_subset, current_data=curr_subset)

    if save_html:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(report_path))
        suite.save_html(str(test_suite_path))

    report_dict = report.as_dict()
    dataset_drift_metric = report_dict["metrics"][0]["result"]
    drift_table_metric = report_dict["metrics"][1]["result"]

    drift_by_columns = drift_table_metric.get("drift_by_columns", {})
    flagged_features = [
        col for col, data in drift_by_columns.items() if data.get("drift_detected", False)
    ]

    return {
        "dataset_drift": dataset_drift_metric.get("dataset_drift", False),
        "drift_share": dataset_drift_metric.get("share_of_drifted_columns", 0.0),
        "drift_share_threshold": drift_share_threshold,
        "number_of_columns": dataset_drift_metric.get("number_of_columns", len(features)),
        "number_of_drifted_columns": dataset_drift_metric.get(
            "number_of_drifted_columns", len(flagged_features)
        ),
        "flagged_features": flagged_features,
        "drift_by_columns": drift_by_columns,
        "report_html_path": str(report_path),
        "test_suite_html_path": str(test_suite_path),
        "test_suite_summary": suite.as_dict().get("summary", {}),
    }


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def benchmark_df() -> pd.DataFrame:
    """Benchmark dataset loaded from data folder."""
    return load_benchmark_data(sample_size=BENCHMARK_SAMPLE_SIZE, random_state=RANDOM_SEED)


@pytest.fixture(scope="module")
def sample_current_df(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Sample current dataset with controlled drift in selected features."""
    return get_or_create_sample_dataset()


@pytest.fixture(scope="module")
def clean_current_df(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Clean reference sample from benchmark with no induced drift."""
    return load_benchmark_data(sample_size=SAMPLE_DATASET_SIZE, random_state=999)


@pytest.fixture(scope="module")
def drift_results(benchmark_df: pd.DataFrame, sample_current_df: pd.DataFrame) -> dict[str, Any]:
    """Execute drift monitoring and return structured results."""
    return run_drift_monitoring(
        reference_df=benchmark_df,
        current_df=sample_current_df,
        features=MONITORED_FEATURES,
        drift_share_threshold=DRIFT_SHARE_THRESHOLD,
        save_html=True,
    )


# ---------------------------------------------------------------------------
# Pytest Tests
# ---------------------------------------------------------------------------
def test_benchmark_data_is_available(benchmark_df: pd.DataFrame) -> None:
    """Verify benchmark data is properly loaded and contains expected columns."""
    assert len(benchmark_df) > 0, "Benchmark dataset is empty"
    for feature in MONITORED_FEATURES:
        assert feature in benchmark_df.columns, f"Benchmark missing expected feature '{feature}'"


def test_sample_dataset_is_available(sample_current_df: pd.DataFrame) -> None:
    """Verify the sample dataset is available in data/ and has matching schema."""
    assert len(sample_current_df) > 0, "Sample current dataset is empty"
    assert (
        SAMPLE_DRIFTED_PARQUET.exists()
    ), f"Sample parquet file missing at {SAMPLE_DRIFTED_PARQUET}"
    assert SAMPLE_DRIFTED_CSV.exists(), f"Sample CSV file missing at {SAMPLE_DRIFTED_CSV}"
    for feature in MONITORED_FEATURES:
        assert feature in sample_current_df.columns, f"Sample dataset missing feature '{feature}'"


def test_evidently_html_report_generated(drift_results: dict[str, Any]) -> None:
    """Verify that human-readable Evidently HTML reports are generated and valid."""
    report_file = Path(drift_results["report_html_path"])
    suite_file = Path(drift_results["test_suite_html_path"])

    assert report_file.exists(), f"Evidently Report HTML not found at {report_file}"
    assert suite_file.exists(), f"Evidently Test Suite HTML not found at {suite_file}"

    # Verify report files contain substantial HTML content
    assert report_file.stat().st_size > 100_000, "Report HTML file is suspiciously small"
    assert suite_file.stat().st_size > 100_000, "Test suite HTML file is suspiciously small"

    html_content = report_file.read_text(encoding="utf-8")
    assert "<html" in html_content.lower(), "Report file does not contain valid HTML"
    assert "Data Drift" in html_content or "evidently" in html_content.lower()


def test_flag_features_exceeding_15_percent_drift(drift_results: dict[str, Any]) -> None:
    """Verify that features with induced drift (>15% shift) are flagged.

    'Average Fare', 'PurchaseLeadTime', and 'Cabin' were intentionally shifted
    and must be detected and flagged by Evidently.
    """
    flagged = set(drift_results["flagged_features"])
    expected_drifted = {"Average Fare", "PurchaseLeadTime", "Cabin"}

    assert expected_drifted.issubset(flagged), (
        f"Expected {expected_drifted} to be flagged for drift exceeding threshold, "
        f"but flagged features were: {flagged}"
    )


def test_stable_features_not_flagged(drift_results: dict[str, Any]) -> None:
    """Verify that unperturbed features with < 15% drift remain unflagged."""
    flagged = set(drift_results["flagged_features"])
    stable_features = {"PAX Count", "Group Status", "DOW", "Farebrand"}

    for feat in stable_features:
        assert feat not in flagged, (
            f"Feature '{feat}' was unexpectedly flagged for drift (score: "
            f"{drift_results['drift_by_columns'].get(feat, {}).get('drift_score')})"
        )


def test_dataset_level_drift_flagged_when_share_exceeds_15_percent(
    drift_results: dict[str, Any],
) -> None:
    """Verify dataset-level drift is flagged when share of drifted features >= 15%."""
    share = drift_results["drift_share"]
    assert share >= DRIFT_SHARE_THRESHOLD, (
        f"Expected drift share >= {DRIFT_SHARE_THRESHOLD}, got {share:.2f}"
    )
    assert drift_results["dataset_drift"] is True, (
        "Dataset drift flag should be True when drifted columns exceed 15%"
    )


def test_no_drift_flagged_on_clean_benchmark_data(
    benchmark_df: pd.DataFrame, clean_current_df: pd.DataFrame
) -> None:
    """Verify that comparing clean benchmark data to itself yields no false drift flags."""
    clean_results = run_drift_monitoring(
        reference_df=benchmark_df,
        current_df=clean_current_df,
        features=MONITORED_FEATURES,
        drift_share_threshold=DRIFT_SHARE_THRESHOLD,
        save_html=False,
    )
    assert clean_results["dataset_drift"] is False, (
        f"Clean data triggered false dataset drift: {clean_results['flagged_features']}"
    )
    assert len(clean_results["flagged_features"]) == 0, (
        f"Clean data falsely flagged features: {clean_results['flagged_features']}"
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint for interactive execution & visual report generation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PAL Passenger Data Pipeline - Evidently AI Data Drift Monitor")
    print("=" * 70)

    print("\n[1/4] Loading benchmark data from data/ folder...")
    ref_data = load_benchmark_data(sample_size=BENCHMARK_SAMPLE_SIZE)
    print(f"      Benchmark reference loaded: {len(ref_data):,} rows")

    print("\n[2/4] Loading / generating sample current dataset...")
    curr_data = get_or_create_sample_dataset()
    print(f"      Current sample loaded: {len(curr_data):,} rows")

    threshold_pct = DRIFT_SHARE_THRESHOLD * 100
    print(f"\n[3/4] Running Evidently drift monitoring (Threshold: {threshold_pct:.0f}%)...")
    results = run_drift_monitoring(
        reference_df=ref_data,
        current_df=curr_data,
        features=MONITORED_FEATURES,
        drift_share_threshold=DRIFT_SHARE_THRESHOLD,
        save_html=True,
    )

    print("\n[4/4] Drift Monitoring Results Summary:")
    print("-" * 70)
    print(f"{'Feature Name':<22} | {'Drift Score':<12} | {'Test Method':<24} | {'Status'}")
    print("-" * 70)

    for col, detail in results["drift_by_columns"].items():
        score = detail.get("drift_score", 0.0)
        method = detail.get("stattest_name", "N/A")
        drifted = detail.get("drift_detected", False)
        status = "[FLAGGED - DRIFT DETECTED]" if drifted else "[PASSED - STABLE]"
        print(f"{col:<22} | {score:<12.4f} | {method:<24} | {status}")

    print("-" * 70)
    drift_pct = results["drift_share"] * 100
    is_drifted = "YES (DRIFT DETECTED)" if results["dataset_drift"] else "NO (STABLE)"
    print(f"Total Monitored Features  : {results['number_of_columns']}")
    print(f"Flagged Drifted Features : {results['number_of_drifted_columns']} ({drift_pct:.1f}%)")
    print(f"Dataset Drift Flag (>15%): {is_drifted}")
    print(f"Flagged Feature List     : {results['flagged_features']}")
    print("\nInteractive HTML Reports Generated for Review:")
    print(f" -> Data Drift Report   : {results['report_html_path']}")
    print(f" -> Test Suite Report   : {results['test_suite_html_path']}")
    print("=" * 70)
