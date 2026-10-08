"""
Quantitative Model Evaluation Script.
Part of Member 2 (ML Detection Lead) deliverables.

Evaluates trained detection models against the ground-truth benchmark labels
and reports Precision, Recall, F1-Score, Accuracy, ROC-AUC, and Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
)
from src.detection.infer import _load_config
from src.detection.baseline import load_isolation_forest, load_kmeans, isolation_forest_score, kmeans_score
from src.detection.autoencoder import load_autoencoder, autoencoder_score


def evaluate_models(npz_path: str = "data/processed/embeddings.npz"):
    data = np.load(npz_path)
    y_true = data["labels"]
    embeddings = data["embeddings"]

    cfg = _load_config()
    thresholds = cfg["thresholds"]

    iso = load_isolation_forest()
    km = load_kmeans()
    ae = load_autoencoder("src/models/autoencoder.pkl")

    # Raw scores
    iso_scores = isolation_forest_score(iso, embeddings)
    km_scores = kmeans_score(km, embeddings)
    ae_scores = autoencoder_score(ae, embeddings)

    # Binary predictions
    iso_pred = (iso_scores >= thresholds["isolation_forest"]).astype(int)
    km_pred = (km_scores >= thresholds["kmeans"]).astype(int)
    ae_pred = (ae_scores >= thresholds["autoencoder"]).astype(int)
    ens_pred = ((iso_pred + km_pred + ae_pred) >= 2).astype(int)

    models = {
        "Isolation Forest (Baseline)": (iso_pred, iso_scores),
        "Autoencoder (PCA)": (ae_pred, ae_scores),
        "K-Means Clustering": (km_pred, km_scores),
        "Ensemble (Majority Vote)": (ens_pred, iso_scores),
    }

    total = len(y_true)
    n_normal = int((y_true == 0).sum())
    n_anomaly = int((y_true == 1).sum())

    print("=" * 85)
    print("      AI-DRIVEN INFRASTRUCTURE MONITORING - MODEL PERFORMANCE REPORT")
    print("=" * 85)
    print(f"Dataset Split: {total:,} logs total | {n_normal:,} Normal (95.9%) | {n_anomaly:,} Anomaly (4.1%)")
    print("-" * 85)
    print(f"{'Model Name':<28} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Accuracy':<8} | {'ROC-AUC':<8}")
    print("-" * 85)

    for name, (pred, scores) in models.items():
        p = precision_score(y_true, pred, zero_division=0) * 100
        r = recall_score(y_true, pred, zero_division=0) * 100
        f1 = f1_score(y_true, pred, zero_division=0) * 100
        acc = accuracy_score(y_true, pred) * 100
        try:
            auc = roc_auc_score(y_true, scores) * 100
        except Exception:
            auc = 0.0
        cm = confusion_matrix(y_true, pred)
        print(f"{name:<28} | {p:>8.2f}% | {r:>7.2f}% | {f1:>7.2f}% | {acc:>7.2f}% | {auc:>7.2f}%")
        print(f"   -> [Confusion Matrix] True Negatives: {cm[0,0]:,}, False Positives: {cm[0,1]:,}, False Negatives: {cm[1,0]:,}, True Positives: {cm[1,1]:,}")

    print("=" * 85)


if __name__ == "__main__":
    evaluate_models()
