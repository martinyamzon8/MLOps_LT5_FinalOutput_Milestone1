from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Paths & Settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "PAL_Passenger_Segmenter")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
ARTIFACT_OUTPUT_PATH = MODELS_DIR / f"pal_passenger_segmenter_{MODEL_VERSION}.joblib"


def setup_mlflow_tracking() -> str:
    """Configure MLflow tracking URI, falling back to local SQLite/directory if server is down."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

    # If it's an HTTP URI, test connection with a short 1s timeout
    if mlflow_uri.startswith("http://") or mlflow_uri.startswith("https://"):
        try:
            req = urllib.request.Request(
                f"{mlflow_uri}/health",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            urllib.request.urlopen(req, timeout=1.0)
            mlflow.set_tracking_uri(mlflow_uri)
            print(f"Connected to MLflow Tracking Server at {mlflow_uri}")
            return mlflow_uri
        except Exception:
            local_db = PROJECT_ROOT / "mlflow.db"
            local_uri = f"sqlite:///{local_db.as_posix()}"
            mlflow.set_tracking_uri(local_uri)
            print(f"MLflow server at {mlflow_uri} not reachable; using local SQLite: {local_uri}")
            return local_uri
    else:
        mlflow.set_tracking_uri(mlflow_uri)
        return mlflow_uri


def load_training_data() -> pd.DataFrame:
    """Load the latest processed parquet artifact or fallback to benchmark data."""
    processed_files = sorted((DATA_DIR / "processed").glob("pal_passengers_clean_*.parquet"))
    if processed_files:
        data_path = processed_files[-1]
        print(f"Loading latest processed data from {data_path}")
    elif (DATA_DIR / "processed" / "pal_passengers_clean_latest.parquet").exists():
        data_path = DATA_DIR / "processed" / "pal_passengers_clean_latest.parquet"
        print(f"Loading data from {data_path}")
    elif (DATA_DIR / "benchmark_data.parquet").exists():
        data_path = DATA_DIR / "benchmark_data.parquet"
        print(f"Loading baseline data from {data_path}")
    else:
        err_msg = "No training data found in data/processed/ or data/benchmark_data.parquet"
        raise FileNotFoundError(err_msg)

    df = pd.read_parquet(data_path)
    return df


def train_and_register_model() -> tuple[KMeans, float, float]:
    """Train KMeans passenger segmentation model, log metrics, and persist artifact."""
    setup_mlflow_tracking()
    df = load_training_data()

    # Standardize column name if needed
    if "Average Fare" in df.columns:
        df.rename(columns={"Average Fare": "AverageFare"}, inplace=True)

    # Select numerical features for clustering
    X = df[["PurchaseLeadTime", "PAX Count", "AverageFare"]].copy()
    X = X.fillna({"PurchaseLeadTime": 0, "PAX Count": 1, "AverageFare": 0})

    # MLflow experiment setup
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pal_passenger_segmentation")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Define 3 Hyperparameters
        n_clusters = 3  # e.g., Leisure, Business, Advance Booking/Group
        init_method = "k-means++"
        max_iterations = 300
        random_state = 42

        # Log Parameters
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("init_method", init_method)
        mlflow.log_param("max_iter", max_iterations)
        mlflow.log_param("random_state", random_state)

        # Train the Model
        model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            max_iter=max_iterations,
            random_state=random_state,
        )
        cluster_labels = model.fit_predict(X)

        # Calculate and Log 2 Clustering Metrics
        sample_size = min(5000, len(X))
        sil_score = float(
            silhouette_score(X, cluster_labels, sample_size=sample_size, random_state=42)
        )
        db_score = float(davies_bouldin_score(X, cluster_labels))

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("davies_bouldin_index", db_score)

        # Set Tags
        mlflow.set_tag("version", MODEL_VERSION)
        mlflow.set_tag("registered_model_name", MODEL_NAME)

        # Log Model to MLflow
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                pip_requirements=["scikit-learn", "pandas", "numpy", "joblib"],
            )

            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if versions:
                latest_ver = sorted(versions, key=lambda v: int(v.version))[-1].version
                client.set_model_version_tag(
                    name=MODEL_NAME,
                    version=latest_ver,
                    key="version_tag",
                    value=MODEL_VERSION,
                )
                print(f"Successfully registered model version {latest_ver} for '{MODEL_NAME}'.")
        except Exception as e:
            print(f"Note: Registry tagging notice: {e}")

        # Always save versioned artifact to disk for standalone endpoint serving
        joblib.dump(model, ARTIFACT_OUTPUT_PATH)
        print(f"Persisted versioned model artifact to {ARTIFACT_OUTPUT_PATH}")

        print(
            f"\nTraining Complete!\n"
            f"Run ID: {run.info.run_id}\n"
            f"Model Name: {MODEL_NAME} | Version: {MODEL_VERSION}\n"
            f"Silhouette Score: {sil_score:.4f} | Davies-Bouldin Index: {db_score:.4f}\n"
        )

        return model, sil_score, db_score


if __name__ == "__main__":
    train_and_register_model()
