"""
Main Interface for Member 3 Deliverables.

Provides straightforward functions to take Member 2's Contract A.2 (DetectionResult)
and generate Member 4's Contract A.3 (IncidentReport).
"""

from typing import List, Optional
from src.schemas import DetectionResult, IncidentReport
from src.llm.client import GroqIncidentReasoner


_shared_reasoner: Optional[GroqIncidentReasoner] = None


def get_reasoner(api_key: Optional[str] = None, model: Optional[str] = None) -> GroqIncidentReasoner:
    """Returns a singleton or configured GroqIncidentReasoner instance."""
    global _shared_reasoner
    if _shared_reasoner is None or api_key is not None or model is not None:
        _shared_reasoner = GroqIncidentReasoner(api_key=api_key, model=model)
    return _shared_reasoner


def diagnose_anomaly(
    detection_result: DetectionResult,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    metrics: Optional[dict] = None
) -> IncidentReport:
    """
    Core function for Member 3 -> Member 4 handoff.
    
    Parameters
    ----------
    detection_result : DetectionResult
        Contract A.2 object produced by Member 2.
    api_key : str, optional
        Groq API key (if None, reads from GROQ_API_KEY environment variable).
    model : str, optional
        Groq model name (defaults to 'llama-3.3-70b-versatile').
    metrics : dict, optional
        Hardware metrics at incident time.
        
    Returns
    -------
    IncidentReport
        Validated Contract A.3 schema object ready for Member 4's dashboard.
    """
    reasoner = get_reasoner(api_key=api_key, model=model)
    return reasoner.diagnose(
        log_id=detection_result.log_id,
        context_window=detection_result.context_window,
        anomaly_score=detection_result.anomaly_score,
        threshold_used=detection_result.threshold_used,
        metrics=metrics
    )


def diagnose_batch(
    detection_results: List[DetectionResult],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    limit: Optional[int] = None
) -> List[IncidentReport]:
    """
    Diagnoses multiple detected anomalies and returns a list of IncidentReport objects.
    Filters automatically for `is_anomaly == True`.
    """
    anomalies = [d for d in detection_results if d.is_anomaly]
    if limit:
        anomalies = anomalies[:limit]

    reports = []
    reasoner = get_reasoner(api_key=api_key, model=model)
    for anomaly in anomalies:
        report = reasoner.diagnose(
            log_id=anomaly.log_id,
            context_window=anomaly.context_window,
            anomaly_score=anomaly.anomaly_score,
            threshold_used=anomaly.threshold_used
        )
        reports.append(report)

    return reports

