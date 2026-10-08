import os
from pathlib import Path
import numpy as np
import joblib
import json
from .data_loader import load_preprocessed
from .baseline import train_isolation_forest, train_kmeans
from .autoencoder import train_autoencoder, save_autoencoder
from .threshold import select_threshold

# Directory to store trained models (shared with baseline)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = MODEL_DIR / "config.json"

def _save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def _load_embeddings(limit: int = None):
    df = load_preprocessed(limit=limit)
    # Expect an "embedding" column containing list[float]
    embeddings = np.stack(df["embedding"].values)
    return embeddings, df

def train_all(limit: int = None, contamination: float = 0.05, n_clusters: int = 10, ae_latent: int = 32, ae_epochs: int = 30):
    """Train IsolationForest, KMeans, and AutoEncoder on the pre‑processed embeddings.

    The trained models are persisted under ``src/models/`` and a small ``config.json``
    containing the chosen thresholds is written alongside them.
    """
    embeddings, _ = _load_embeddings(limit)
    # --- Baselines ---
    iso = train_isolation_forest(embeddings, contamination=contamination)
    kmeans = train_kmeans(embeddings, n_clusters=n_clusters)
    # --- Auto‑Encoder ---
    ae = train_autoencoder(embeddings, n_components=ae_latent)
    save_autoencoder(ae, MODEL_DIR / "autoencoder.pkl")
    # Determine thresholds (simple percentile on training scores)
    from .baseline import isolation_forest_score, kmeans_score
    iso_scores = isolation_forest_score(iso, embeddings)
    km_scores = kmeans_score(kmeans, embeddings)
    # Auto‑Encoder scores (reconstruction error)
    from .autoencoder import autoencoder_score
    ae_scores = autoencoder_score(ae, embeddings)
    # Choose thresholds at 95th percentile of each score distribution
    iso_thr = select_threshold(iso_scores, percentile=95)
    km_thr = select_threshold(km_scores, percentile=95)
    ae_thr = select_threshold(ae_scores, percentile=95)
    # Save raw and normalized thresholds
    def _norm(val, s_min, s_max):
        return float(np.clip((val - s_min) / (s_max - s_min + 1e-8), 0.0, 1.0))

    stats = {
        "isolation_forest": {"min": float(iso_scores.min()), "max": float(iso_scores.max())},
        "kmeans": {"min": float(km_scores.min()), "max": float(km_scores.max())},
        "autoencoder": {"min": float(ae_scores.min()), "max": float(ae_scores.max())},
    }

    norm_thresholds = {
        "isolation_forest": _norm(iso_thr, iso_scores.min(), iso_scores.max()),
        "kmeans": _norm(km_thr, km_scores.min(), km_scores.max()),
        "autoencoder": _norm(ae_thr, ae_scores.min(), ae_scores.max()),
    }

    config = {
        "thresholds": {
            "isolation_forest": iso_thr,
            "kmeans": km_thr,
            "autoencoder": ae_thr,
        },
        "norm_thresholds": norm_thresholds,
        "stats": stats,
        "contamination": contamination,
        "n_clusters": n_clusters,
        "ae_latent_dim": ae_latent,
    }
    _save_config(config)
    print("Training complete. Models and config saved to", MODEL_DIR)

if __name__ == "__main__":
    # Simple CLI entry point
    import argparse
    parser = argparse.ArgumentParser(description="Train detection models")
    parser.add_argument("--limit", type=int, default=None, help="Load only first N rows for quick dev runs")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--ae_latent", type=int, default=32)
    parser.add_argument("--ae_epochs", type=int, default=30)
    args = parser.parse_args()
    train_all(limit=args.limit, contamination=args.contamination, n_clusters=args.n_clusters, ae_latent=args.ae_latent, ae_epochs=args.ae_epochs)
