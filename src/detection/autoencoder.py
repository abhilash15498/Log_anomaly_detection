import joblib
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

# Simple PCA‑based auto‑encoder replacement

def train_autoencoder(embeddings: np.ndarray, n_components: int = 32) -> PCA:
    """Fit a PCA model to the embeddings.

    The principal components act as the encoder; reconstruction from the components
    provides a low‑dimensional approximation. The reconstruction error (MSE) is used
    as the anomaly score.
    """
    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    pca.fit(embeddings)
    return pca

def save_autoencoder(model: PCA, path: Path):
    """Persist the trained PCA model using ``joblib``."""
    joblib.dump(model, path)

def load_autoencoder(path: Path) -> PCA:
    """Load a previously saved PCA‑based auto‑encoder."""
    return joblib.load(path)

def autoencoder_score(model: PCA, embeddings: np.ndarray) -> np.ndarray:
    """Compute reconstruction‑error (MSE) scores for each sample.

    Higher scores indicate a larger deviation from the learned subspace → more
    anomalous.
    """
    # Project to latent space and back
    transformed = model.transform(embeddings)
    recon = model.inverse_transform(transformed)
    mse = np.mean((embeddings - recon) ** 2, axis=1)
    return mse
