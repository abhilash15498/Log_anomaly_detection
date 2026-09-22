# AI-Driven Infrastructure Monitoring and Intelligent Incident Recommendation System

> **Course**: BAI506 Mini Project — Academic Year 2026–27  
> **Department**: Artificial Intelligence and Machine Learning, BMSIT&M  
> **Batch**: C3 | **Guide**: Dr. Archana Bhat  

---

## 👥 Team & Role Allocation

| Role | Team Member | USN | Core Responsibilities |
|---|---|---|---|
| **Member 1 — Data & Preprocessing Lead** | Abhilash Hiremath | `1BY24AI002` | Dataset acquisition (HDFS/BGL), Docker fault-injection app, Drain3 parsing, sentence-transformers embeddings (`src/ingestion`, `src/preprocessing`, `data/`) |
| **Member 2 — ML Detection Lead** | Shrikrishna R Prabhu | `1BY24AI157` | Unsupervised baseline (Isolation Forest / K-Means), Autoencoder (PyTorch), threshold tuning & evaluation (`src/detection`) |
| **Member 3 — LLM / RAG Lead** | Deepak Suresh Naik | `1BY24AI037` | Context aggregation window, structured prompt design, LLM integration (Ollama / API), optional RAG store (`src/llm`) |
| **Member 4 — Integration & Dashboard Lead** | Vishnu Karanth A | `1BY24AI191` | Streamlit incident dashboard, administrator feedback loop, end-to-end pipeline wiring, integration tests (`src/dashboard`, `tests/`) |

---

📖 **Quick Reference**: See the full [Team Coordination & Step-by-Step Build Guide](file:///c:/Users/hmabh/OneDrive/Desktop/Mini%20Project/docs/TEAM_COORDINATION_GUIDE.md) for beginner-friendly explanations, step-by-step milestones, and handoff contracts.

---

## 📌 Project Overview

Modern cloud and enterprise IT infrastructures continuously generate massive volumes of logs and performance metrics. Traditional threshold and keyword monitoring approaches often produce excessive false alarms or fail to catch subtle anomaly patterns. Moreover, when an anomaly is flagged, administrators must manually piece together log context to deduce the root cause and remediation steps.

This project introduces a **two-stage architecture** designed to keep continuous monitoring lightweight while delivering actionable operational insights:
1. **Stage 1 (Lightweight Anomaly Detection)**: Continuous unsupervised ML models (Isolation Forest / Autoencoder) score incoming preprocessed logs and metrics in real time with minimal resource overhead.
2. **Stage 2 (LLM Contextual Reasoning & Remediation)**: When an anomaly crosses the threshold, surrounding log context and metrics are aggregated and dispatched to a Large Language Model (or RAG pipeline). The LLM generates a human-readable explanation, root-cause hypothesis, and concrete mitigation steps.
3. **Feedback Loop**: System administrators review flagged incidents on an interactive Streamlit dashboard. Confirming or dismissing alerts feeds back into threshold tuning to mitigate alert fatigue.

---

## 📁 Repository Structure & Module Ownership

```text
project-root/
|-- data/
|   |-- raw/                  # Untouched downloads (HDFS, BGL)     [Member 1]
|   `-- processed/            # Embeddings + cleaned logs          [Member 1]
|-- src/
|   |-- __init__.py
|   |-- schemas.py            # Pydantic data contracts (A.1 - A.4) [Shared / M4]
|   |-- ingestion/            # Log/metric ingest & fault injector [Member 1]
|   |-- preprocessing/        # Log parsing & sentence-embeddings  [Member 1]
|   |-- detection/            # Baseline & Autoencoder models      [Member 2]
|   |-- llm/                  # Prompts, LLM client, optional RAG  [Member 3]
|   `-- dashboard/            # Streamlit app & pipeline glue      [Member 4]
|-- notebooks/                # Exploration & prototyping (EDA)    [All]
|-- tests/                    # Unit & end-to-end integration test [M4 coordinates]
|-- docs/                     # Synopsis, build guide, notes       [All]
|-- requirements.txt
|-- .gitignore
`-- README.md                 # Setup & documentation reference   [All]
```

---

## 🔄 Data Contracts & Interface Schemas

The four pipeline workstreams communicate strictly via standardized schemas defined in [`src/schemas.py`](file:///c:/Users/hmabh/OneDrive/Desktop/Mini%20Project/src/schemas.py):

### 1. Preprocessed Log Record (`Member 1` → `Member 2`)
```json
{
  "log_id": "string — unique id for this log line",
  "timestamp": "string — ISO 8601",
  "raw_text": "string — original log line",
  "source": "string — service / host name",
  "embedding": [0.012, -0.045, 0.089, "..."],
  "metrics": {
    "cpu_percent": 98.4,
    "memory_percent": 95.2
  }
}
```

### 2. Detection Result (`Member 2` → `Members 3 & 4`)
```json
{
  "log_id": "string",
  "anomaly_score": 0.842,
  "is_anomaly": true,
  "threshold_used": 0.750,
  "context_window": ["log_id_prev_2", "log_id_prev_1", "log_id", "log_id_next_1"]
}
```

### 3. Incident Report (`Member 3` → `Member 4`)
```json
{
  "log_id": "string",
  "explanation": "Out of memory error triggered by heap allocation spike in worker thread.",
  "severity": "critical",
  "recommended_actions": [
    "Increase JVM max heap space allocation (-Xmx).",
    "Inspect worker queue leak in task executor."
  ],
  "retrieved_context": ["Similar past incident INC-2024-089 (Heap leak in Worker-3)"]
}
```

### 4. Feedback Record (`Member 4` → `Member 2`)
```json
{
  "log_id": "string",
  "administrator_verdict": "confirmed", 
  "timestamp": "2026-09-22T19:00:00Z"
}
```

---

## 🛠️ Technology Stack

| Layer | Technology / Library |
|---|---|
| **Language** | Python 3.10+ |
| **Datasets** | LogHub (HDFS_v1, BGL) + Dockerized fault-injection environment |
| **Log Parsing** | Drain3 / regex tokenizers |
| **Log Representation** | `sentence-transformers` (`all-MiniLM-L6-v2`) or TF-IDF |
| **Anomaly Detection** | `scikit-learn` (Isolation Forest, K-Means), `PyTorch` (Autoencoder) |
| **LLM & Reasoning** | Ollama (Local Llama/Mistral) or API-based LLM |
| **Vector Search (RAG)** | FAISS / ChromaDB (Optional stretch goal) |
| **Dashboard UI** | Streamlit |
| **Environment / Testing**| Docker, Docker Compose, `pytest` |

---

## 🎯 Definition of Done

The project is complete and ready for submission/viva when:
- [ ] **Quantitative Evaluation**: Precision, Recall, F1-score, and Accuracy reported on held-out test splits for both baseline (Isolation Forest) and deep (Autoencoder) models.
- [ ] **False-Positive Rate**: FPR measured and reported on normal-only holdout periods.
- [ ] **Fault-Injection Showcase**: Live demo with at least 3 distinct injected fault types producing detector alerts and LLM explanation/remediation outputs.
- [ ] **Interactive Dashboard**: Streamlit interface displaying real-time alerts, LLM explanations, and administrator confirm/dismiss controls.
- [ ] **Team Readiness**: Every team member can explain the entire end-to-end pipeline.

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd "Mini Project"

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline Components
- **Data Ingestion & Preprocessing**: `python -m src.preprocessing.pipeline`
- **Model Training & Detection**: `python -m src.detection.train`
- **Dashboard**: `streamlit run src/dashboard/app.py`
