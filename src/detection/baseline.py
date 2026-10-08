import os
import joblib
from pathlib import Path
from typing import Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Directory to store trained models
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_isolation_forest(embeddings: np.ndarray, contamination: float = 0.05) -> IsolationForest:
    """Train an IsolationForest on the provided embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        2‑D array of shape (n_samples, n_features).
    contamination : float, optional
        Expected proportion of outliers in the data (default 0.05).
    """
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(embeddings)
    # Persist model
    joblib.dump(iso, MODEL_DIR / "isolation_forest.joblib")
    return iso

def train_kmeans(embeddings: np.ndarray, n_clusters: int = 10) -> KMeans:
    """Train a K‑Means clustering model.

    The distance of a point to its nearest cluster centre is used as an
    anomaly score (larger distance ⇒ more anomalous).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(embeddings)
    joblib.dump(kmeans, MODEL_DIR / "kmeans.joblib")
    return kmeans

def load_isolation_forest() -> IsolationForest:
    return joblib.load(MODEL_DIR / "isolation_forest.joblib")

def load_kmeans() -> KMeans:
    return joblib.load(MODEL_DIR / "kmeans.joblib")

def isolation_forest_score(model: IsolationForest, embeddings: np.ndarray) -> np.ndarray:
    """Return anomaly scores (the lower, the more normal) from IsolationForest.
    We invert the sign to make higher scores mean more anomalous.
    """
    raw = model.decision_function(embeddings)  # negative values = outliers
    return -raw

def kmeans_score(model: KMeans, embeddings: np.ndarray) -> np.ndarray:
    """Compute distance to nearest centroid as anomaly score.
    """
    distances = model.transform(embeddings)  # shape (n_samples, n_clusters)
    return distances.min(axis=1)
