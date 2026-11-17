"""
Unit Tests for Member 4 (Integration & Dashboard Lead).

Verifies Schema Contract A.4 (FeedbackRecord), feedback persistence,
dynamic threshold adaptation, and pipeline integration.
"""

from pathlib import Path
import pytest
from src.schemas import FeedbackRecord
from src.dashboard.feedback import save_feedback, load_feedback, get_feedback_summary, adapt_threshold, persist_adapted_thresholds


def test_feedback_record_contract_a4(tmp_path):
    temp_file = tmp_path / "test_feedback.jsonl"
    record = save_feedback(
        log_id="log_0000409",
        verdict="confirmed",
        feedback_file=temp_file
    )

    assert isinstance(record, FeedbackRecord)
    assert record.log_id == "log_0000409"
    assert record.administrator_verdict == "confirmed"
    assert record.timestamp is not None
    assert temp_file.exists()


def test_feedback_summary_and_persistence(tmp_path):
    temp_file = tmp_path / "test_feedback.jsonl"
    save_feedback("log_1", "confirmed", feedback_file=temp_file)
    save_feedback("log_2", "dismissed", feedback_file=temp_file)
    save_feedback("log_3", "dismissed", feedback_file=temp_file)

    summary = get_feedback_summary(feedback_file=temp_file)
    assert summary["total"] == 3
    assert summary["confirmed"] == 1
    assert summary["dismissed"] == 2


def test_dynamic_threshold_adaptation():
    base = 0.50

    # Case 1: Multiple dismissals (false alarms) -> threshold increases to reduce noise
    dismissals = [
        FeedbackRecord(log_id=f"log_{i}", administrator_verdict="dismissed")
        for i in range(5)
    ]
    adapted_higher = adapt_threshold(base, feedback_records=dismissals, learning_rate=0.02)
    assert adapted_higher > base

    # Case 2: Multiple confirmations (real incidents) -> threshold stays sensitive
    confirmations = [
        FeedbackRecord(log_id=f"log_{i}", administrator_verdict="confirmed")
        for i in range(5)
    ]
    adapted_lower = adapt_threshold(base, feedback_records=confirmations, learning_rate=0.02)
    assert adapted_lower < base


def test_persist_adapted_thresholds(tmp_path):
    import json
    cfg_file = tmp_path / "test_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump({"consensus_cutoff": 0.50}, f)

    fb_file = tmp_path / "feedback.jsonl"
    save_feedback("log_test", "dismissed", feedback_file=fb_file)

    new_cutoff = persist_adapted_thresholds(config_path=cfg_file, feedback_file=fb_file, learning_rate=0.05)
    assert new_cutoff is not None
    assert new_cutoff > 0.50

    with open(cfg_file, "r", encoding="utf-8") as f:
        saved_cfg = json.load(f)
    assert saved_cfg["consensus_cutoff"] == new_cutoff


def test_integration_pipeline_wiring():
    from src.detection.infer import detect
    from src.llm.diagnose import diagnose_anomaly

    # Test end-to-end integration: Member 2 detector -> Member 3 LLM -> Member 4 consumable
    results = detect(limit=50)
    assert len(results) == 50

    # Ensure Contract A.2 feeds cleanly into Contract A.3
    sample_anomaly = next((r for r in results if r.is_anomaly), None)
    if sample_anomaly:
        report = diagnose_anomaly(sample_anomaly)
        assert report.log_id == sample_anomaly.log_id
        assert report.severity in ["low", "medium", "high", "critical"]
        assert len(report.recommended_actions) >= 1

