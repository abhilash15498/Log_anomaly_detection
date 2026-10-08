import json
import warnings
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from ..schemas import DetectionResult
from .data_loader import load_preprocessed
from .baseline import load_isolation_forest, load_kmeans, isolation_forest_score, kmeans_score
from .autoencoder import load_autoencoder, autoencoder_score

# Silence Scikit-learn unpickle warnings
warnings.filterwarnings("ignore")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "models" / "config.json"


def _load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Detection config not found at {CONFIG_PATH}. Run train.py first.")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_score(score: float, s_min: float, s_max: float) -> float:
    """Scale raw model score into a normalized 0.0 - 1.0 score."""
    denom = s_max - s_min
    if abs(denom) < 1e-8:
        return 0.5
    norm = (score - s_min) / denom
    return float(np.clip(norm, 0.0, 1.0))


def _extract_context(df: pd.DataFrame, idx: int, window: int = 2) -> List[str]:
    """Return a list of raw log texts surrounding the record at idx."""
    start = max(0, idx - window)
    end = min(len(df), idx + window + 1)
    return df.iloc[start:end]["raw_text"].tolist()


def detect(limit: Optional[int] = None, model_name: str = "ensemble") -> List[DetectionResult]:
    """Run detection models on pre-processed data and return DetectionResult objects.

    Parameters
    ----------
    limit : int, optional
        Number of logs to analyze (default all).
    model_name : str, default "ensemble"
        One of "ensemble", "isolation_forest", "kmeans", or "autoencoder".
    """
    df = load_preprocessed(limit=limit)
    embeddings = np.stack(df["embedding"].values)

    cfg = _load_config()
    thresholds = cfg["thresholds"]
    stats = cfg.get("stats", {})
    norm_thresholds = cfg.get("norm_thresholds", {})

    iso = load_isolation_forest()
    km = load_kmeans()
    ae = load_autoencoder(Path(__file__).resolve().parents[1] / "models" / "autoencoder.pkl")

    # Compute raw scores
    iso_raw = isolation_forest_score(iso, embeddings)
    km_raw = kmeans_score(km, embeddings)
    ae_raw = autoencoder_score(ae, embeddings)

    # Compute normalized scores in [0.0, 1.0]
    iso_min = stats.get("isolation_forest", {}).get("min", float(iso_raw.min()))
    iso_max = stats.get("isolation_forest", {}).get("max", float(iso_raw.max()))
    km_min = stats.get("kmeans", {}).get("min", float(km_raw.min()))
    km_max = stats.get("kmeans", {}).get("max", float(km_raw.max()))
    ae_min = stats.get("autoencoder", {}).get("min", float(ae_raw.min()))
    ae_max = stats.get("autoencoder", {}).get("max", float(ae_raw.max()))

    iso_norm = np.clip((iso_raw - iso_min) / (iso_max - iso_min + 1e-8), 0.0, 1.0)
    km_norm = np.clip((km_raw - km_min) / (km_max - km_min + 1e-8), 0.0, 1.0)
    ae_norm = np.clip((ae_raw - ae_min) / (ae_max - ae_min + 1e-8), 0.0, 1.0)

    iso_thr_norm = norm_thresholds.get("isolation_forest", _normalize_score(thresholds["isolation_forest"], iso_min, iso_max))
    km_thr_norm = norm_thresholds.get("kmeans", _normalize_score(thresholds["kmeans"], km_min, km_max))
    ae_thr_norm = norm_thresholds.get("autoencoder", _normalize_score(thresholds["autoencoder"], ae_min, ae_max))

    results: List[DetectionResult] = []
    for i, row in df.iterrows():
        iso_flag = bool(iso_raw[i] >= thresholds["isolation_forest"])
        km_flag = bool(km_raw[i] >= thresholds["kmeans"])
        ae_flag = bool(ae_raw[i] >= thresholds["autoencoder"])

        if model_name == "isolation_forest":
            is_anomaly = iso_flag
            score = float(iso_norm[i])
            thr = float(iso_thr_norm)
        elif model_name == "autoencoder":
            is_anomaly = ae_flag
            score = float(ae_norm[i])
            thr = float(ae_thr_norm)
        elif model_name == "kmeans":
            is_anomaly = km_flag
            score = float(km_norm[i])
            thr = float(km_thr_norm)
        else:  # Ensemble (Majority voting: requires at least 2 models to agree, preventing false alarms)
            votes = int(iso_flag) + int(km_flag) + int(ae_flag)
            is_anomaly = bool(votes >= 2)
            # Max normalized score across detectors
            score = float(max(iso_norm[i], km_norm[i], ae_norm[i]))
            # Report average normalized cutoff threshold for consistent scale
            thr = float((iso_thr_norm + km_thr_norm + ae_thr_norm) / 3.0)

        result = DetectionResult(
            log_id=row["log_id"],
            anomaly_score=round(score, 4),
            is_anomaly=is_anomaly,
            threshold_used=round(thr, 4),
            context_window=_extract_context(df, i, window=2),
        )
        results.append(result)

    return results
