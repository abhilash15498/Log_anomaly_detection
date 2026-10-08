import numpy as np
from typing import Union

def select_threshold(scores: Union[np.ndarray, list], percentile: float = 95.0) -> float:
    """Select a score threshold based on a percentile of the training scores.

    Parameters
    ----------
    scores : array‑like
        The anomaly scores computed on the training data.
    percentile : float, default 95.0
        The percentile of the score distribution to use as the cut‑off. A higher
        percentile yields a stricter (higher) threshold, reducing false‑positives
        but possibly missing subtle anomalies.
    """
    if isinstance(scores, list):
        scores = np.array(scores)
    if scores.ndim != 1:
        raise ValueError("Scores array must be 1‑dimensional.")
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100.")
    return float(np.percentile(scores, percentile))
