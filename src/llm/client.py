"""
Groq LLM Client & Reasoning Engine for Member 3.

Connects to the Groq Cloud API for ultra-fast inference (<0.5s),
enforces structured JSON output matching Schema Contract A.3, and includes
a reliable offline fallback engine for internet-less demo/testing environments.
"""

import os
import json
import re
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from src.schemas import IncidentReport
from src.llm.prompts import SYSTEM_PROMPT, build_analysis_prompt

_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.is_file():
    load_dotenv(dotenv_path=_env_file)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class GroqIncidentReasoner:
    """
    LLM Incident Reasoning Engine powered by Groq API.
    """

    DEFAULT_MODEL = "openai/gpt-oss-20b"
    FALLBACK_MODEL = "qwen/qwen3.8-27b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self._client = None

        if self.api_key and GROQ_AVAILABLE:
            try:
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                self._client = None

    @property
    def is_connected(self) -> bool:
        """Returns True if a live Groq API connection is configured."""
        return self._client is not None

    def diagnose(
        self,
        log_id: str,
        context_window: List[str],
        anomaly_score: float = 0.5,
        threshold_used: float = 0.5,
        metrics: Optional[dict] = None
    ) -> IncidentReport:
        """
        Diagnoses an anomaly event using Groq LLM (or fallback heuristic if offline).
        Returns a validated Schema Contract A.3 IncidentReport.
        """
        user_prompt = build_analysis_prompt(
            log_id=log_id,
            context_window=context_window,
            anomaly_score=anomaly_score,
            threshold_used=threshold_used,
            metrics=metrics
        )

        if self._client:
            try:
                return self._call_groq_api(log_id, user_prompt)
            except Exception as e:
                pass  # Fall back gracefully

        return self._fallback_diagnosis(log_id, context_window, anomaly_score)

    def _call_groq_api(self, log_id: str, user_prompt: str) -> IncidentReport:
        """Calls Groq Chat Completion and extracts valid JSON."""
        chat_completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=600
        )

        content = chat_completion.choices[0].message.content or ""
        data = self._extract_json(content)

        raw_sev = str(data.get("severity", "medium")).lower()
        if "critical" in raw_sev:
            severity = "critical"
        elif "high" in raw_sev:
            severity = "high"
        elif "low" in raw_sev:
            severity = "low"
        else:
            severity = "medium"

        raw_actions = data.get("recommended_actions", [])
        if isinstance(raw_actions, list) and raw_actions:
            actions = [str(a) for a in raw_actions]
        elif isinstance(raw_actions, str):
            actions = [raw_actions]
        else:
            actions = [
                "Inspect related microservice logs surrounding this timestamp.",
                "Verify cluster resource utilization and network reachability."
            ]

        explanation = data.get("explanation")
        if not explanation or not isinstance(explanation, str):
            explanation = "Operational anomaly detected by ML detection engine in log stream."

        return IncidentReport(
            log_id=log_id,
            explanation=explanation.strip(),
            severity=severity,
            recommended_actions=actions,
            retrieved_context=None
        )

    def _extract_json(self, text: str) -> dict:
        """Extracts JSON object from text or markdown blocks."""
        text = text.strip()
        # Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Match ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # Match first { to last }
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        return {}

    def _fallback_diagnosis(
        self,
        log_id: str,
        context_window: List[str],
        anomaly_score: float
    ) -> IncidentReport:
        """
        Deterministic, offline SRE heuristic fallback engine.
        Ensures Contract A.3 compliance even when offline or without an API key.
        """
        combined_text = " ".join(context_window).lower()

        if "outofmemoryerror" in combined_text or "heap space" in combined_text:
            return IncidentReport(
                log_id=log_id,
                explanation="Critical JVM memory exhaustion: Application worker exceeded configured Java Heap Space during object allocation.",
                severity="critical",
                recommended_actions=[
                    "Increase JVM heap allocation flags: -Xmx4g -Xms2g in container startup arguments.",
                    "Capture and inspect heap dump with: jmap -dump:live,format=b,file=heap.hprof <pid>",
                    "Restart affected worker pod or service instance."
                ],
                retrieved_context=["Known Pattern: JVM_HEAP_EXHAUSTION_ERR"]
            )
        elif "hikaripool" in combined_text or "connection timeout" in combined_text or "pool exhausted" in combined_text:
            return IncidentReport(
                log_id=log_id,
                explanation="Database connection pool starvation: Microservice threads were unable to acquire a SQL connection within timeout.",
                severity="high",
                recommended_actions=[
                    "Scale HikariCP maximumPoolSize from current limit to 50 in application.properties.",
                    "Check for long-running unindexed SQL queries blocking active connections.",
                    "Verify PostgreSQL/MySQL database server max_connections headroom."
                ],
                retrieved_context=["Known Pattern: DB_CONNECTION_POOL_STARVATION"]
            )
        elif "nullpointerexception" in combined_text or "status=500" in combined_text or "circuit breaker" in combined_text:
            return IncidentReport(
                log_id=log_id,
                explanation="Cascading application error: Unhandled null reference triggered downstream HTTP 500 error propagation and tripped circuit breaker.",
                severity="high",
                recommended_actions=[
                    "Inspect stack trace in microservice logs for missing payload null-checks.",
                    "Check circuit breaker health endpoint (/actuator/health).",
                    "Roll back recent microservice canary deployment if error rate persists."
                ],
                retrieved_context=["Known Pattern: UNHANDLED_RUNTIME_EXCEPTION"]
            )
        elif "allocateblock" in combined_text or "dfs" in combined_text or "block" in combined_text:
            return IncidentReport(
                log_id=log_id,
                explanation="Distributed filesystem storage block allocation anomaly: Unusual block replica request rate or delayed block report acknowledged by NameNode.",
                severity="medium",
                recommended_actions=[
                    "Run 'hdfs fsck /' to check distributed filesystem block health and replica placement.",
                    "Inspect DataNode network throughput and verify dfs.datanode.failed.volumes.tolerated.",
                    "Review concurrent MapReduce or Spark writer job scheduling."
                ],
                retrieved_context=["Known Pattern: HDFS_BLOCK_ALLOCATION_DEVIATION"]
            )
        else:
            return IncidentReport(
                log_id=log_id,
                explanation=f"Statistical anomaly flagged by ML detection ensemble with deviation score {anomaly_score:.4f}.",
                severity="medium",
                recommended_actions=[
                    "Review surrounding logs across microservices for correlated errors.",
                    "Verify server CPU, memory, and disk I/O metrics around incident timestamp."
                ],
                retrieved_context=None
            )
