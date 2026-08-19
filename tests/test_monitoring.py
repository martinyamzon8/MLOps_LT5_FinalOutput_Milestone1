"""Tests for Evidently drift monitoring module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.run_report import (  # noqa: E402
    attach_predictions,
    generate_drift_report,
    load_datasets,
    load_or_train_model,
)


def test_load_datasets():
    """Verify load_datasets returns non-empty reference and current dataframes."""
    ref_df, curr_df = load_datasets()
    assert len(ref_df) > 0
    assert len(curr_df) > 0
    assert "AverageFare" in ref_df.columns
    assert "AverageFare" in curr_df.columns


def test_load_or_train_model():
    """Verify model loading / training returns a valid KMeans instance."""
    ref_df, _ = load_datasets()
    model = load_or_train_model(ref_df)
    assert hasattr(model, "predict")
    assert hasattr(model, "cluster_centers_")
    assert len(model.cluster_centers_) == 3


def test_attach_predictions():
    """Verify predictions and confidence scores are correctly attached to dataframe."""
    ref_df, _ = load_datasets()
    model = load_or_train_model(ref_df)
    sample = ref_df.head(10).copy()
    with_preds = attach_predictions(sample, model)
    assert "prediction" in with_preds.columns
    assert "confidence" in with_preds.columns
    assert len(with_preds) == 10
    assert set(with_preds["prediction"].unique()).issubset({0, 1, 2})
    assert (with_preds["confidence"] >= 0.0).all()
    assert (with_preds["confidence"] <= 1.0).all()


def test_generate_drift_report(tmp_path, monkeypatch):
    """Verify full drift report execution creates HTML output."""
    out_html = generate_drift_report()
    assert out_html.exists()
    assert out_html.stat().st_size > 1000
