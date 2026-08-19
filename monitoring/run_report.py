"""Evidently AI Drift Monitoring Pipeline.

Generates the Evidently HTML report comparing reference vs current production data,
covering data drift and target/prediction drift.

Saved to: reports/evidently_report.html
Runnable via:
    python monitoring/run_report.py
    or: uv run python monitoring/run_report.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Evidently imports with backward/forward compatibility
try:
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report
except ImportError:
    from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

BENCHMARK_PATH = DATA_DIR / "benchmark_data.parquet"
DRIFTED_PATH = DATA_DIR / "sample_drifted_data.parquet"
REPORT_OUTPUT_PATH = REPORTS_DIR / "evidently_report.html"
DEFAULT_MODEL_PATH = MODELS_DIR / "pal_passenger_segmenter_v1.0.0.joblib"

DRIFT_SHARE_THRESHOLD = 0.15  # 15% drift share threshold
SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference baseline dataset and current production dataset."""
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found at {BENCHMARK_PATH}")
    if not DRIFTED_PATH.exists():
        raise FileNotFoundError(f"Current dataset not found at {DRIFTED_PATH}")

    ref_df = pd.read_parquet(BENCHMARK_PATH)
    curr_df = pd.read_parquet(DRIFTED_PATH)

    # Standardize column naming if needed
    for df in [ref_df, curr_df]:
        if "Average Fare" in df.columns and "AverageFare" not in df.columns:
            df["AverageFare"] = df["Average Fare"]

    # Sample for snappy report generation
    if len(ref_df) > SAMPLE_SIZE:
        ref_df = ref_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(curr_df) > SAMPLE_SIZE:
        curr_df = curr_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    return ref_df, curr_df


def load_or_train_model(ref_df: pd.DataFrame) -> KMeans:
    """Load trained clustering model or fit baseline on reference data."""
    if DEFAULT_MODEL_PATH.exists():
        model = joblib.load(DEFAULT_MODEL_PATH)
    else:
        # Fallback to any joblib in models/
        joblib_files = sorted(MODELS_DIR.glob("*.joblib"))
        if joblib_files:
            model = joblib.load(joblib_files[-1])
        else:
            X = ref_df[["PurchaseLeadTime", "PAX Count", "AverageFare"]].fillna(
                {"PurchaseLeadTime": 0, "PAX Count": 1, "AverageFare": 0}
            )
            model = KMeans(n_clusters=3, random_state=RANDOM_SEED).fit(X)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, DEFAULT_MODEL_PATH)
    return model


def attach_predictions(df: pd.DataFrame, model: KMeans) -> pd.DataFrame:
    """Attach cluster predictions and distance-based confidence scores to DataFrame."""
    out = df.copy()
    X = out[["PurchaseLeadTime", "PAX Count", "AverageFare"]].fillna(
        {"PurchaseLeadTime": 0, "PAX Count": 1, "AverageFare": 0}
    )
    predictions = model.predict(X)
    out["prediction"] = predictions

    # Calculate distance-based confidence
    distances = model.transform(X)
    min_dist = distances.min(axis=1)
    mean_dist = np.maximum(distances.mean(axis=1), 1e-5)
    out["confidence"] = (1.0 / (1.0 + (min_dist / mean_dist))).round(4)
    return out


def generate_drift_report() -> Path:
    """Generate and save Evidently HTML drift report covering data and target drift."""
    print("=" * 70)
    print("PHILIPPINE AIRLINES MLOPS: EVIDENTLY DRIFT MONITORING REPORT")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets
    ref_df, curr_df = load_datasets()
    print(f"Loaded reference dataset: {len(ref_df)} rows")
    print(f"Loaded current dataset:   {len(curr_df)} rows")

    # 2. Attach model predictions for target/prediction drift
    model = load_or_train_model(ref_df)
    ref_with_preds = attach_predictions(ref_df, model)
    curr_with_preds = attach_predictions(curr_df, model)

    # 3. Setup column mapping
    num_features = ["AverageFare", "PurchaseLeadTime", "PAX Count"]
    cat_features = [
        col
        for col in ["Cabin", "Farebrand", "DOW", "POS Region", "Ticketing Channel", "Group Status"]
        if col in ref_df.columns
    ]

    col_mapping = ColumnMapping(
        prediction="prediction",
        numerical_features=num_features,
        categorical_features=cat_features,
        task="classification",
    )

    # 4. Build Evidently Report with Data Drift and Target/Prediction Drift presets
    report = Report(
        metrics=[
            DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD),
            TargetDriftPreset(),
        ]
    )

    print("\nRunning statistical drift tests across features and model predictions...")
    report.run(
        reference_data=ref_with_preds,
        current_data=curr_with_preds,
        column_mapping=col_mapping,
    )

    # 5. Save HTML report
    report.save_html(str(REPORT_OUTPUT_PATH))
    print(f"\nEvidently drift report successfully saved to:\n  -> {REPORT_OUTPUT_PATH.resolve()}")

    # 6. Extract summary metrics for console output
    try:
        report_dict = report.as_dict()
        data_drift_metric = report_dict["metrics"][0]["result"]
        share_drifted = data_drift_metric.get("share_of_drifted_columns", 0.0)
        drift_by_columns = data_drift_metric.get("drift_by_columns", {})

        print("\n" + "-" * 75)
        print("SUMMARY OF STATISTICAL DRIFT MONITORING RESULTS")
        print("-" * 75)
        print(f"Dataset Drift Detected:   {data_drift_metric.get('dataset_drift', False)}")
        print(
            f"Share of Drifted Columns: {share_drifted:.1%} "
            f"(Threshold: {DRIFT_SHARE_THRESHOLD:.1%})"
        )
        print("\nFeature-by-Feature Statistical Breakdown:")
        for col_name, info in drift_by_columns.items():
            drifted = info.get("drift_detected", False)
            score = info.get("drift_score", 0.0)
            stattest = info.get("stattest_name", "Statistical test")
            flag = ">>> DRIFT DETECTED <<<" if drifted else "STABLE"
            print(f"  - {col_name:18s} | Score: {score:.4f} | Method: {stattest:26s} | {flag}")
        print("-" * 75)
    except Exception as e:
        print(f"Note: Console summary formatting notice: {e}")

    return REPORT_OUTPUT_PATH


if __name__ == "__main__":
    generate_drift_report()
