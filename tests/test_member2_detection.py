"""
Unit tests for Member 2 (ML Detection Lead) pipeline.
Tests baseline models, autoencoder, thresholding, and Schema Contract A.2 compliance.
"""

import pytest
from src.schemas import DetectionResult
from src.detection.infer import detect


def test_detection_contract_compliance():
    results = detect(limit=50)
    assert len(results) == 50

    for res in results:
        assert isinstance(res, DetectionResult)
        assert res.log_id.startswith("log_")
        assert 0.0 <= res.anomaly_score <= 1.0
        assert 0.0 <= res.threshold_used <= 1.0
        assert isinstance(res.is_anomaly, bool)
        assert isinstance(res.context_window, list)
        assert len(res.context_window) >= 1


def test_detection_anomaly_rate_reasonable():
    results = detect(limit=200)
    anomalies = [r for r in results if r.is_anomaly]
    anomaly_rate = len(anomalies) / len(results)

    # Anomaly rate should not be 100% or 0%
    assert 0.0 < anomaly_rate < 0.25, f"Anomaly rate was {anomaly_rate * 100:.1f}%, expected 1-25%"


def test_detection_context_window():
    results = detect(limit=20)
    # Check middle item has full context window
    mid_item = results[5]
    assert len(mid_item.context_window) == 5  # 2 prev + 1 center + 2 next
