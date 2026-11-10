"""
Prompt Engineering & Templates for Member 3 (LLM Reasoning Lead).

Translates Schema Contract A.2 (DetectionResult with 5-log context window)
into a structured prompt for Groq LLM and enforces JSON output matching
Schema Contract A.3 (IncidentReport).
"""

from typing import List, Optional


SYSTEM_PROMPT = """You are an expert Principal Site Reliability Engineer (SRE) and Cloud Infrastructure Architect.
Your task is to analyze server logs flagged by an automated machine learning anomaly detector and produce an actionable Incident Report.

You will be given:
1. An anomaly log ID and score.
2. A chronological 5-log context window (surrounding logs before the incident, the specific anomalous line marked with '>>', and subsequent logs).
3. Optional hardware metrics (CPU/RAM).

Your diagnostic output MUST strictly follow this JSON structure:
{
  "explanation": "A concise, technical, plain-English summary explaining the root cause and why this pattern is anomalous.",
  "severity": "low" | "medium" | "high" | "critical",
  "recommended_actions": [
    "Specific command or action 1",
    "Specific command or action 2",
    "Specific command or action 3"
  ]
}

Severity Guidelines:
- "critical": Complete service outage, data corruption, OutOfMemoryError, disk/node death.
- "high": Cascade failures, connection pool exhaustion, high packet loss, repeated timeouts.
- "medium": Elevated latency, replica re-routing, transient network reset, degraded capacity.
- "low": Non-blocking anomaly, unexpected template drift, routine retry success.

Respond ONLY with valid JSON. Do not include markdown code blocks or explanatory commentary outside the JSON object.
"""


def build_analysis_prompt(
    log_id: str,
    context_window: List[str],
    anomaly_score: float,
    threshold_used: float,
    metrics: Optional[dict] = None
) -> str:
    """
    Constructs the user prompt for the LLM using the 5-log context window.
    """
    formatted_context = "\n".join(
        f"  {line}" for line in context_window
    ) if context_window else "  (No surrounding context available)"

    metrics_str = ""
    if metrics:
        metrics_str = f"\nHardware Metrics at incident: CPU={metrics.get('cpu_percent', 'N/A')}%, RAM={metrics.get('memory_percent', 'N/A')}%\n"

    prompt = f"""[INCIDENT EVALUATION REQUEST]
Target Log ID: {log_id}
Anomaly Score: {anomaly_score:.4f} (Detector Cutoff: {threshold_used:.4f})
{metrics_str}
Chronological Context Window (5 logs):
{formatted_context}

Analyze this incident. Return your assessment strictly in the required JSON format.
"""
    return prompt

