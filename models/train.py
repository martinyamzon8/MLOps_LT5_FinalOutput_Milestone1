import os

import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Set MLflow tracking URI if active server is running
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
mlflow.set_tracking_uri(mlflow_uri)

# 1. Load clean artifact from Milestone 1
df = pd.read_parquet("data/processed/pal_passengers_clean_latest.parquet")

# Standardize column name if needed
if "Average Fare" in df.columns:
    df.rename(columns={"Average Fare": "AverageFare"}, inplace=True)

# Select numerical features for clustering
X = df[["PurchaseLeadTime", "PAX Count", "AverageFare"]].copy()
# Handle missing values (e.g. PurchaseLeadTime has nulls when PNRCreationDate is missing)
X = X.fillna({"PurchaseLeadTime": 0, "PAX Count": 1, "AverageFare": 0})

# 2. Set the Experiment Name
mlflow.set_experiment("pal_passenger_segmentation")

with mlflow.start_run():
    # Define 3 Hyperparameters
    n_clusters = 3  # e.g., Leisure, Business, Group
    init_method = "k-means++"
    max_iterations = 300

    # Log Parameters
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("init_method", init_method)
    mlflow.log_param("max_iter", max_iterations)

    # 3. Train the Model
    model = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        max_iter=max_iterations,
        random_state=42,
    )
    cluster_labels = model.fit_predict(X)

    # 4. Calculate and Log 2 Clustering Metrics
    # Silhouette Score: Measures how distinct passenger segments are (sampling for speed & memory)
    sil_score = float(silhouette_score(X, cluster_labels, sample_size=10000, random_state=42))
    # Davies-Bouldin: Measures cluster separation/dispersion
    db_score = float(davies_bouldin_score(X, cluster_labels))

    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.log_metric("davies_bouldin_index", db_score)

    # 5. Log and Register the Model
    mlflow.set_tag("version", "v1.0.0")
    mlflow.set_tag("registered_model_name", "PAL_Passenger_Segmenter")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="PAL_Passenger_Segmenter",
        pip_requirements=["scikit-learn", "pandas", "numpy"],
    )

    # Tag model version in MLflow Model Registry
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='PAL_Passenger_Segmenter'")
    if versions:
        latest_ver = sorted(versions, key=lambda v: int(v.version))[-1].version
        client.set_model_version_tag(
            name="PAL_Passenger_Segmenter",
            version=latest_ver,
            key="version_tag",
            value="v1.0.0",
        )
        print(f"Successfully registered model version {latest_ver} for 'PAL_Passenger_Segmenter'.")


    print(
        f"Training complete! MLflow experiment: 'pal_passenger_segmentation'\n"
        f"Silhouette Score: {sil_score:.4f} | Davies-Bouldin Index: {db_score:.4f}"
    )
