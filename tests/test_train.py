"""Tests for model training module (models/train.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train import (  # noqa: E402
    load_training_data,
    setup_mlflow_tracking,
    train_and_register_model,
)


def test_setup_mlflow_tracking():
    """Verify MLflow setup returns a valid tracking URI."""
    uri = setup_mlflow_tracking()
    assert isinstance(uri, str)
    assert len(uri) > 0


def test_load_training_data():
    """Verify load_training_data loads non-empty DataFrame with required features."""
    df = load_training_data()
    assert len(df) > 0
    assert "PurchaseLeadTime" in df.columns
    assert "PAX Count" in df.columns


def test_train_and_register_model():
    """Verify train_and_register_model produces fitted model and metrics meeting thresholds."""
    model, sil_score, db_score = train_and_register_model()
    assert hasattr(model, "predict")
    assert sil_score >= 0.45, f"Silhouette score {sil_score} below 0.45"
    assert db_score <= 1.0, f"Davies-Bouldin index {db_score} above 1.0"
