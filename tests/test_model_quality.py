import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

# passenger behavior doesn't sort clearly and 
# may have some features that are the same as others. 
# 0.45 sits comfortably above the 0.25 faint structure line,
# so it catches genuine degradation, but still achievable given real behavioral data. 
# Below this, the segments overlap too much to base targeting decisions on
MIN_SILHOUETTE = 0.45 
# it sits on a recognized boundary between decent and workable data separation. 
# Lower is better, while 0 is perfect.
MAX_DAVIES_BOULDIN = 1.0 
# without a seed, the synthetic data and the model's starting positions differ every run, 
# so the metrics wobble and the test passes or fails at random. 
# Fixing it means a failure signals a real code change, not luck.
RANDOM_SEED = 42


@pytest.fixture(scope="module")
def synthetic_passengers() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    n = 100
    biz_lead = rng.normal(7, 3, n)
    biz_pax = rng.normal(1, 0.3, n)
    biz_fare = rng.normal(12000, 2000, n)
    leisure_lead = rng.normal(30, 10, n)
    leisure_pax = rng.normal(2, 0.5, n)
    leisure_fare = rng.normal(7000, 1000, n)
    grp_lead = rng.normal(30, 10, n)
    grp_pax = rng.normal(10, 3, n)
    grp_fare = rng.normal(10000, 1500, n)
    lead = np.concatenate([biz_lead, leisure_lead, grp_lead])
    pax = np.concatenate([biz_pax, leisure_pax, grp_pax])
    fare = np.concatenate([biz_fare, leisure_fare, grp_fare])
    return pd.DataFrame({
        "PurchaseLeadTime": lead,
        "AverageFare": fare,
        "PAX Count": pax,
    })


def test_clustering_meets_quality_thresholds(synthetic_passengers):
    X = synthetic_passengers[["PurchaseLeadTime", "PAX Count", "AverageFare"]]
    model = KMeans(n_clusters=3, random_state=RANDOM_SEED)
    labels = model.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    assert silhouette >= MIN_SILHOUETTE, (
        f"Silhouette {silhouette:.4f} below minimum {MIN_SILHOUETTE}"
    )
    assert davies_bouldin <= MAX_DAVIES_BOULDIN, (
        f"Davies-Bouldin {davies_bouldin:.4f} above maximum {MAX_DAVIES_BOULDIN}"
    )

def test_cluster_labels_have_valid_format(synthetic_passengers):
    X = synthetic_passengers[["PurchaseLeadTime", "PAX Count", "AverageFare"]]
    model = KMeans(n_clusters=3, random_state=RANDOM_SEED)
    labels = model.fit_predict(X)
    assert len(labels) == len(synthetic_passengers), (
        f"Got {len(labels)} labels for {len(synthetic_passengers)} rows"
    )
    assert set(labels).issubset({0, 1, 2}), f"Unexpected labels: {set(labels)}"
    assert not np.isnan(labels).any(), "Model produced NaN predictions"