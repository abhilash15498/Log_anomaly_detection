"""
Administrator Feedback Loop Handler (Member 4 Deliverable).

Implements Schema Contract A.4 (FeedbackRecord) and dynamic threshold adaptation
based on human administrator verdicts (confirm anomaly vs dismiss false positive).
"""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional
from datetime import datetime, timezone

from src.schemas import FeedbackRecord

FEEDBACK_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "feedback_records.jsonl"


def save_feedback(
    log_id: str,
    verdict: Literal["confirmed", "dismissed"],
    feedback_file: Optional[Path] = None
) -> FeedbackRecord:
    """
    Records an administrator decision for a flagged incident into Contract A.4.
    Appends to feedback_records.jsonl.
    """
    target_path = feedback_file or FEEDBACK_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    record = FeedbackRecord(
        log_id=log_id,
        administrator_verdict=verdict,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")

    return record


def load_feedback(feedback_file: Optional[Path] = None) -> List[FeedbackRecord]:
    """Loads all feedback records from disk."""
    target_path = feedback_file or FEEDBACK_FILE
    if not target_path.exists():
        return []

    records = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(FeedbackRecord.model_validate_json(line.strip()))
                except Exception:
                    continue
    return records


def get_feedback_summary(feedback_file: Optional[Path] = None) -> Dict[str, int]:
    """Returns total counts of confirmed vs dismissed alerts."""
    records = load_feedback(feedback_file)
    confirmed = sum(1 for r in records if r.administrator_verdict == "confirmed")
    dismissed = sum(1 for r in records if r.administrator_verdict == "dismissed")
    return {
        "total": len(records),
        "confirmed": confirmed,
        "dismissed": dismissed
    }


def adapt_threshold(
    base_threshold: float,
    feedback_records: Optional[List[FeedbackRecord]] = None,
    learning_rate: float = 0.02
) -> float:
    """
    Dynamically adapts the sensitivity threshold:
    - If admin dismisses false alarms, threshold increases slightly (reduces false alarm fatigue).
    - If admin confirms anomalies, threshold slightly tightens (increases recall sensitivity).
    """
    records = feedback_records if feedback_records is not None else load_feedback()
    if not records:
        return base_threshold

    # Calculate net false-positive pressure
    dismissed = sum(1 for r in records if r.administrator_verdict == "dismissed")
    confirmed = sum(1 for r in records if r.administrator_verdict == "confirmed")

    net_dismissals = dismissed - confirmed
    adjustment = net_dismissals * learning_rate
    # Bound adaptation within [0.1, 0.95]
    adapted = max(0.1, min(0.95, base_threshold + adjustment))
    return round(adapted, 4)

