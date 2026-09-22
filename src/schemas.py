"""
Interface Schemas & Data Contracts for:
AI-Driven Infrastructure Monitoring and Intelligent Incident Recommendation System

Reference: Project Implementation Plan Appendix A
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PreprocessedLogRecord(BaseModel):
    """
    Contract A.1: Preprocessed Log Record (Member 1 -> Member 2)
    Produced by ingestion & preprocessing pipelines.
    """
    log_id: str = Field(..., description="Unique ID for this log line")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp string")
    raw_text: str = Field(..., description="Original raw log message")
    source: str = Field(..., description="Service or host name (e.g., datanode, auth-service)")
    embedding: List[float] = Field(..., description="Numerical representation vector (e.g., 384-dim MiniLM)")
    metrics: Optional[dict] = Field(
        default=None,
        description="Optional system metrics snapshot (e.g., {'cpu_percent': 98.4, 'memory_percent': 95.2})"
    )


class DetectionResult(BaseModel):
    """
    Contract A.2: Detection Result (Member 2 -> Members 3 & 4)
    Produced by unsupervised anomaly detection models (Isolation Forest / Autoencoder).
    """
    log_id: str = Field(..., description="Log ID matching the evaluated record")
    anomaly_score: float = Field(..., description="Unsupervised anomaly deviation score")
    is_anomaly: bool = Field(..., description="True if anomaly_score >= threshold_used")
    threshold_used: float = Field(..., description="Cutoff threshold applied for this decision")
    context_window: List[str] = Field(
        default_factory=list,
        description="List of surrounding log IDs (temporal context window) for LLM analysis"
    )


class IncidentReport(BaseModel):
    """
    Contract A.3: Incident Report (Member 3 -> Member 4)
    Produced by the LLM reasoning & prompt engineering layer.
    """
    log_id: str = Field(..., description="Log ID of the root anomalous event")
    explanation: str = Field(..., description="Plain-language explanation of the likely cause")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Assessed incident severity")
    recommended_actions: List[str] = Field(..., description="Concrete, actionable remediation steps")
    retrieved_context: Optional[List[str]] = Field(
        default=None,
        description="Optional similar historical incidents retrieved from RAG vector store"
    )


class FeedbackRecord(BaseModel):
    """
    Contract A.4: Feedback Record (Member 4 -> Member 2)
    Captured from the administrator dashboard for threshold tuning.
    """
    log_id: str = Field(..., description="Flagged log ID under review")
    administrator_verdict: Literal["confirmed", "dismissed"] = Field(
        ...,
        description="Administrator verdict: confirmed anomaly or false positive dismissal"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp when feedback was recorded"
    )
