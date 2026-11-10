"""
Unit & Integration Tests for Member 3 (LLM Reasoning Layer).

Verifies Schema Contract A.3 adherence, prompt formatting,
and deterministic fallback reasoning.
"""

import pytest
from src.schemas import DetectionResult, IncidentReport
from src.llm.prompts import build_analysis_prompt
from src.llm.client import GroqIncidentReasoner
from src.llm.diagnose import diagnose_anomaly, diagnose_batch


def test_prompt_formatting():
    context = [
        "2026-10-06T12:00:00Z [INFO] Service started",
        "2026-10-06T12:00:01Z [WARN] Memory elevated",
        ">> 2026-10-06T12:00:02Z [ERROR] java.lang.OutOfMemoryError",
        "2026-10-06T12:00:03Z [FATAL] Worker terminated",
        "2026-10-06T12:00:04Z [INFO] Restarting pod"
    ]
    prompt = build_analysis_prompt(
        log_id="test_log_001",
        context_window=context,
        anomaly_score=0.88,
        threshold_used=0.75,
        metrics={"cpu_percent": 85.0, "memory_percent": 98.5}
    )

    assert "test_log_001" in prompt
    assert "OutOfMemoryError" in prompt
    assert "98.5%" in prompt
    assert "0.8800" in prompt


def test_contract_a3_compliance_oom():
    reasoner = GroqIncidentReasoner(api_key=None)  # Uses offline deterministic engine
    context = [
        "10:00:00 [INFO] Request received",
        ">> 10:00:02 [ERROR] java.lang.OutOfMemoryError: Java heap space",
        "10:00:03 [CRITICAL] Thread died"
    ]

    report = reasoner.diagnose(
        log_id="log_oom_01",
        context_window=context,
        anomaly_score=0.92,
        threshold_used=0.75
    )

    assert isinstance(report, IncidentReport)
    assert report.log_id == "log_oom_01"
    assert report.severity == "critical"
    assert "heap" in report.explanation.lower() or "memory" in report.explanation.lower()
    assert len(report.recommended_actions) >= 1
    assert any("-Xmx" in action or "heap" in action.lower() for action in report.recommended_actions)

    # Validate JSON round-trip
    dumped = report.model_dump_json()
    validated = IncidentReport.model_validate_json(dumped)
    assert validated.log_id == report.log_id


def test_contract_a3_compliance_db_pool():
    reasoner = GroqIncidentReasoner(api_key=None)
    context = [
        ">> [ERROR] HikariPool-1 - Connection is not available, request timed out after 30000ms."
    ]

    report = reasoner.diagnose(
        log_id="log_db_01",
        context_window=context,
        anomaly_score=0.84,
        threshold_used=0.75
    )

    assert report.severity == "high"
    assert "database" in report.explanation.lower() or "connection" in report.explanation.lower()
    assert len(report.recommended_actions) >= 1


def test_diagnose_anomaly_from_contract_a2():
    # Simulate Member 2's Contract A.2 input
    detection = DetectionResult(
        log_id="log_0000409",
        anomaly_score=0.4400,
        is_anomaly=True,
        threshold_used=0.5263,
        context_window=[
            "081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: blk_-2444...",
            ">> 081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: blk_-1942...",
            "081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: blk_4649..."
        ]
    )

    # Member 3 function call
    report = diagnose_anomaly(detection)

    assert isinstance(report, IncidentReport)
    assert report.log_id == "log_0000409"
    assert report.severity in ["low", "medium", "high", "critical"]
    assert len(report.recommended_actions) >= 1
    assert isinstance(report.explanation, str) and len(report.explanation) > 10


def test_diagnose_batch():
    results = [
        DetectionResult(log_id="log_1", anomaly_score=0.2, is_anomaly=False, threshold_used=0.5, context_window=[]),
        DetectionResult(log_id="log_2", anomaly_score=0.8, is_anomaly=True, threshold_used=0.5, context_window=[">> OutOfMemoryError"]),
        DetectionResult(log_id="log_3", anomaly_score=0.7, is_anomaly=True, threshold_used=0.5, context_window=[">> HikariPool timeout"])
    ]

    reports = diagnose_batch(results)
    # Only the 2 anomalies should be diagnosed
    assert len(reports) == 2
    assert reports[0].log_id == "log_2"
    assert reports[1].log_id == "log_3"

