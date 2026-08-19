from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "pal_passenger_segmenter_v1.0.0.joblib"


class ModelService:
    """Manages model loading from MLflow Registry or versioned artifacts, and executes inference."""

    def __init__(self) -> None:
        self.model_name: str = os.getenv("MODEL_NAME", "PAL_Passenger_Segmenter")
        self.model_version: str = os.getenv("MODEL_VERSION", "v1.0.0")
        self.mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        artifact_env = os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_ARTIFACT_PATH))
        self.artifact_path: Path = Path(artifact_env)
        self.model: KMeans | None = None
        self.model_source: str = "unloaded"
        self.cluster_segment_map: dict[int, str] = {}

    def _is_mlflow_available(self) -> bool:
        """Fast check to verify if the MLflow HTTP server is reachable."""
        if not self.mlflow_uri:
            return False
        if self.mlflow_uri.startswith("http://") or self.mlflow_uri.startswith("https://"):
            try:
                url = f"{self.mlflow_uri}/health"
                req = urllib.request.Request(url, headers={"User-Agent": "FastAPI"})
                urllib.request.urlopen(req, timeout=0.5)
                return True
            except Exception:
                return False
        return True

    def load_model(self) -> None:
        """Load model from MLflow registry or versioned disk artifact."""
        loaded_model = None

        # 1. Attempt loading from MLflow Model Registry if server is verified reachable
        if self._is_mlflow_available():
            try:
                import mlflow.sklearn

                mlflow.set_tracking_uri(self.mlflow_uri)
                model_uri = f"models:/{self.model_name}/{self.model_version}"
                loaded_model = mlflow.sklearn.load_model(model_uri)
                self.model_source = f"mlflow_registry ({model_uri})"
                print(f"Loaded model from MLflow Registry: {model_uri}")
            except Exception as e:
                print(f"MLflow registry load failed ({e}); falling back to local artifact.")

        # 2. Fallback to local versioned artifact file
        if loaded_model is None:
            if self.artifact_path.exists():
                loaded_model = joblib.load(self.artifact_path)
                self.model_source = f"artifact ({self.artifact_path.name})"
                print(f"Loaded versioned model artifact from: {self.artifact_path}")
            else:
                # Fallback: search for any .joblib in models/
                joblib_files = sorted((PROJECT_ROOT / "models").glob("*.joblib"))
                if joblib_files:
                    loaded_model = joblib.load(joblib_files[-1])
                    self.model_source = f"artifact_fallback ({joblib_files[-1].name})"
                    print(f"Loaded latest joblib artifact from: {joblib_files[-1]}")
                else:
                    # If no artifact exists, train on benchmark baseline
                    print("No artifact found on disk. Training initial baseline model...")
                    benchmark_path = PROJECT_ROOT / "data" / "benchmark_data.parquet"
                    if benchmark_path.exists():
                        df = pd.read_parquet(benchmark_path)
                        if "Average Fare" in df.columns:
                            df.rename(columns={"Average Fare": "AverageFare"}, inplace=True)
                        X = df[["PurchaseLeadTime", "PAX Count", "AverageFare"]].fillna(
                            {"PurchaseLeadTime": 0, "PAX Count": 1, "AverageFare": 0}
                        )
                        loaded_model = KMeans(n_clusters=3, random_state=42).fit(X)
                        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(loaded_model, self.artifact_path)
                        self.model_source = "trained_baseline_artifact"
                    else:
                        raise RuntimeError("Cannot load ML model: data and artifacts missing.")

        self.model = loaded_model
        self._build_segment_map()

    def _build_segment_map(self) -> None:
        """Map cluster IDs to human-readable business segments."""
        if self.model is None or not hasattr(self.model, "cluster_centers_"):
            self.cluster_segment_map = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}
            return

        centers = self.model.cluster_centers_  # shape (n_clusters, 3)
        n_clusters = len(centers)

        segment_map: dict[int, str] = {}
        if n_clusters == 3:
            # Centroid features: [PurchaseLeadTime, PAX Count, AverageFare]
            fare_sorted = np.argsort(centers[:, 2])
            lead_sorted = np.argsort(centers[:, 0])

            highest_fare_cluster = int(fare_sorted[-1])
            longest_lead_cluster = int(lead_sorted[-1])

            if longest_lead_cluster == highest_fare_cluster:
                longest_lead_cluster = int(lead_sorted[-2])

            remaining = [
                c for c in range(3) if c not in (highest_fare_cluster, longest_lead_cluster)
            ]
            standard_leisure_cluster = remaining[0] if remaining else 0

            segment_map[highest_fare_cluster] = "Business / Premium Traveler"
            segment_map[longest_lead_cluster] = "Advance Booking / Group Travel"
            segment_map[standard_leisure_cluster] = "Standard Leisure Traveler"
        else:
            for i in range(n_clusters):
                segment_map[i] = f"Market Segment {i + 1}"

        self.cluster_segment_map = segment_map

    def predict(
        self,
        purchase_lead_time: float,
        pax_count: int,
        average_fare: float,
    ) -> dict[str, Any]:
        """Run inference and return prediction, confidence score, and segment label."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Prepare feature vector: [PurchaseLeadTime, PAX Count, AverageFare]
        input_data = pd.DataFrame(
            [
                {
                    "PurchaseLeadTime": float(purchase_lead_time),
                    "PAX Count": int(pax_count),
                    "AverageFare": float(average_fare),
                }
            ]
        )

        cluster_id = int(self.model.predict(input_data)[0])

        # Compute confidence score using distance to cluster centers
        distances = self.model.transform(input_data)[0]
        scale = float(np.mean(distances)) if float(np.mean(distances)) > 0 else 1.0
        logits = -distances / scale
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        confidence = float(np.round(probabilities[cluster_id], 4))

        confidence = max(0.0, min(1.0, confidence))
        segment_name = self.cluster_segment_map.get(cluster_id, f"Segment {cluster_id}")

        return {
            "prediction": cluster_id,
            "segment_name": segment_name,
            "confidence": confidence,
            "model_version": self.model_version,
            "features": {
                "PurchaseLeadTime": purchase_lead_time,
                "PAX Count": pax_count,
                "AverageFare": average_fare,
            },
        }


# Global singleton instance
model_service = ModelService()
