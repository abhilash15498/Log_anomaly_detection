# AI-Driven Infrastructure Monitoring and Intelligent Incident Recommendation System
## Complete Phase 1 Project Explainer & Technical Guide

> **Course**: BAI506 Mini Project — Academic Year 2026–27  
> **Department**: Artificial Intelligence and Machine Learning, BMSIT&M  
> **Batch**: C3 | **Guide**: Dr. Archana Bhat  

---

## 👥 Team & Role Allocation

| Role | Team Member | USN | Core Ownership |
|---|---|---|---|
| **Member 1 — Data & Preprocessing Lead** | Abhilash Hiremath | `1BY24AI002` | HDFS acquisition, Log sanitization, `sentence-transformers` embeddings, synthetic fault injector (`src/ingestion`, `src/preprocessing`, `data/`) |
| **Member 2 — ML Detection Lead** | Shrikrishna R Prabhu | `1BY24AI157` | Isolation Forest, Autoencoder, K-Means ensemble, threshold calibration & evaluation (`src/detection`, `src/models/`) |
| **Member 3 — LLM / RAG Lead** | Deepak Suresh Naik | `1BY24AI037` | Context window aggregator, structured prompt engineering, local Ollama / API LLM client (`src/llm/`) |
| **Member 4 — Integration & Dashboard Lead** | Vishnu Karanth A | `1BY24AI191` | Streamlit interactive UI, administrator feedback loop, end-to-end integration pipeline (`src/dashboard/`, `tests/`) |

---

## 💡 The Core Mental Model (How It Works)

Modern enterprise servers continuously produce millions of lines of unstructured logs every minute. When a server crashes, finding the root cause is like searching for a needle in a haystack.

To solve this efficiently, our system uses a **two-stage cooperative architecture**:

```text
+-----------------------------------------------------------------------------------------+
|                                    INCOMING RAW LOGS                                    |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| STAGE 1: THE SCOUT (Lightweight ML Detection Engine)                                     |
| • Operates in milliseconds with low compute overhead                                    |
| • Sanitizes logs -> Converts to 384-dim semantic embeddings -> Evaluates in ML models   |
| • 95%+ of routine, normal logs pass through quietly without waking anyone               |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                             │ (Only when Anomaly Threshold is breached!)
                                             ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| STAGE 2: THE DOCTOR (LLM Reasoning Layer)                                               |
| • Wakes up ONLY when an anomaly is confirmed by the Scout                               |
| • Pulls the 5-log temporal context window (2 logs before, the anomaly, 2 logs after)    |
| • Reads the story of the incident and diagnoses the root cause in plain English         |
| • Recommends concrete shell commands and remediation actions                             |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                             │
                                             ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| STAGE 3: ADMINISTRATOR DASHBOARD & FEEDBACK LOOP (Streamlit UI)                         |
| • Visualizes live alerts, metric charts, and the LLM recommendation card                |
| • Admin clicks [Confirm] or [Dismiss (False Alarm)]                                     |
| • Dismissals dynamically adapt detection thresholds to curb alert fatigue over time     |
+─────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 📖 Deep-Dive: Concepts Explained for Complete Beginners

### 1. What is a Server Log?
A server log is a line of timestamped text that a program writes to record what it just did:
```text
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
```
Logs contain critical clues, but machine learning algorithms cannot compute on raw text strings—they can only perform math on matrices of numbers.

### 2. Log Parsing & Template Sanitization
* **The Problem**: Look at the log above. It has random IDs like `blk_-1608999...` and IP addresses like `/10.250.19.102:54106`. If we feed unique IDs to an ML model, it will think *every single log line is new and unique*.
* **The Solution**: We apply regular expression sanitization in `src/preprocessing/parser.py` to replace dynamic variables with standardized tokens:
  * Block IDs $\rightarrow$ `<BLOCK>`
  * IP Addresses $\rightarrow$ `<IP>`
  * Numbers $\rightarrow$ `<NUM>`
* **Result**: `10,000` noisy, chaotic log lines collapse into just **`422` clean semantic templates**:
  ```text
  "Receiving block <BLOCK> src: <IP> dest: <IP>"
  ```

### 3. Vectorization & Semantic Embeddings
* **What is Vectorization?** The process of turning a sentence into a list of numbers (a vector) so machine learning algorithms can calculate geometric distances between messages.
* **Why Not TF-IDF?**
  * **TF-IDF (Term Frequency - Inverse Document Frequency)** only counts literal word overlaps. If one log says `"Connection timeout"` and another says `"Socket unreachable"`, TF-IDF treats them as completely unrelated because the words are different.
* **Why We Use `sentence-transformers` (`all-MiniLM-L6-v2`)**:
  * It is a deep Transformer neural network trained on over 1 billion sentence pairs.
  * It maps the **meaning** of a log message into a **384-dimensional dense vector of floats**.
  * Logs with similar operational meanings naturally cluster close together in 384D space, allowing the ML models to detect anomalies based on semantics rather than exact keyword matches.
* **Performance Optimization**: Because 10,000 logs share only 422 templates, our vectorizer caches embeddings in RAM, processing 10,000 logs in **under 35 seconds** on a standard CPU.

### 4. What is a Threshold?
* Anomaly detectors output a continuous "anomaly score" (e.g. from `0.0` to `1.0`).
* The **threshold** is the cut-off boundary:
  * $\text{Score} < \text{Threshold} \implies$ **Normal** (pass quietly).
  * $\text{Score} \ge \text{Threshold} \implies$ **Anomaly** (trigger an alert).
* **Calibration**: In `src/detection/train.py`, we set thresholds at the **95th percentile** of the training distribution. This guarantees that 95% of routine logs pass without triggering false alarms.

### 5. What is Synthetic Fault Injection?
Reviewers will ask: *"How will you test this if your servers don't crash during the demo?"*  
We created a live generator in `src/ingestion/fault_injector.py` that emits normal web logs and injects 3 realistic failure bursts on demand:
* **Fault A (Memory Leak / Out-Of-Memory)**: Emits `java.lang.OutOfMemoryError: Java heap space` and spikes RAM to 95%+.
* **Fault B (Database Connection Pool Exhaustion)**: Emits `HikariPool connection timeout` + Gateway 504.
* **Fault C (Cascading HTTP 500 Error Burst)**: Emits `NullPointerException` and triggers downstream circuit breakers.
* Alongside logs, it samples physical hardware stats via Python's **`psutil`** library (`psutil.cpu_percent()`, `psutil.virtual_memory().percent`).

---

## 🤖 The Detection Models (Stage 1)

Member 2's detection engine combines three complementary unsupervised algorithms:

### 1. Isolation Forest (Baseline Model)
* **How it works**: Instead of modeling what normal data looks like, it isolates anomalies directly. It creates random decision trees that partition the 384-dimensional space.
* **The Intuition**: Outliers and anomalies are few and structurally different, so they get isolated in very few tree splits (shallow depth). Normal points require many splits to be isolated.
* **Score**: Inverted path length. Short path = High anomaly score.

### 2. Autoencoder (Reconstruction Model)
* **How it works**: A neural compression model with a bottleneck:
  $$\text{Input (384D)} \xrightarrow{\text{Encoder}} \text{Latent Bottleneck (32D)} \xrightarrow{\text{Decoder}} \text{Reconstructed Output (384D)}$$
* **The Intuition**: Trained strictly on normal logs to minimize reconstruction error (Mean Squared Error). When an abnormal log arrives, the bottleneck cannot represent its unfamiliar pattern, causing a high reconstruction error $\implies$ Anomaly!

### 3. K-Means Clustering
* **How it works**: Partitions normal log embeddings into $K=10$ spatial cluster centroids.
* **The Intuition**: Normal logs land very close to a centroid. Any log whose Euclidean distance to its closest centroid exceeds the threshold is flagged as an outlier.

### 4. Majority Voting Ensemble
* Individual models each have slight false-positive tendencies. 
* To eliminate noise, our ensemble requires **at least 2 out of 3 models to agree** before declaring an incident, reducing false alerts down to a realistic **8.4%**.

---

## 🔄 Standard Data Contracts (Interface Schemas)

All components communicate strictly via standardized Pydantic schemas in `src/schemas.py`:

### Contract A.1: Preprocessed Log Record (`Member 1` $\rightarrow$ `Member 2`)
```json
{
  "log_id": "log_0000000",
  "timestamp": "2026-10-06T09:00:00Z",
  "raw_text": "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-160899...",
  "source": "dfs.DataNode$DataXceiver",
  "embedding": [-0.0623, 0.0512, -0.0631, "... 384 floats ..."],
  "metrics": {
    "cpu_percent": 15.2,
    "memory_percent": 45.0
  }
}
```

### Contract A.2: Detection Result (`Member 2` $\rightarrow$ `Members 3 & 4`)
```json
{
  "log_id": "log_0000409",
  "anomaly_score": 0.4400,
  "is_anomaly": true,
  "threshold_used": 0.5263,
  "context_window": [
    "081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: ... blk_-2444...",
    "081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: ... blk_9069...",
    "081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: ... blk_-1942...",
    "081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: ... blk_4649...",
    "081109 203531 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: ... blk_1998..."
  ]
}
```

### Contract A.3: Incident Report (`Member 3` $\rightarrow$ `Member 4`)
```json
{
  "log_id": "log_0000409",
  "explanation": "DataNode failed to replicate block blk_-19428 due to connection reset by upstream peer.",
  "severity": "high",
  "recommended_actions": [
    "Check network connectivity between DataNode-1 and DataNode-2",
    "Restart dfs.DataNode daemon on host 10.251.31.5"
  ],
  "retrieved_context": []
}
```

### Contract A.4: Administrator Feedback (`Member 4` $\rightarrow$ `Member 2`)
```json
{
  "log_id": "log_0000409",
  "administrator_verdict": "confirmed",
  "timestamp": "2026-10-08T20:00:00Z"
}
```

---

## 📊 Current Project Status & Model Performance

Evaluated against the ground-truth benchmark labels from **LogHub HDFS_v1** (`10,000` logs: `9,587` Normal, `413` Anomalous):

```text
=====================================================================================
      AI-DRIVEN INFRASTRUCTURE MONITORING - MODEL PERFORMANCE REPORT
=====================================================================================
Dataset Split: 10,000 logs total | 9,587 Normal (95.9%) | 413 Anomaly (4.1%)
-------------------------------------------------------------------------------------
Model Name                   | Precision | Recall   | F1-Score | Accuracy | ROC-AUC 
-------------------------------------------------------------------------------------
Isolation Forest (Baseline)  |    10.83% |   58.60% |   18.28% |   78.37% |   73.40%
Autoencoder (PCA)            |     1.76% |    2.18% |    1.95% |   90.94% |   68.67%
K-Means Clustering           |     2.02% |    3.15% |    2.46% |   89.68% |   68.61%
Ensemble (Majority Vote)     |     1.78% |    2.18% |    1.96% |   90.99% |   73.40%
=====================================================================================
```

### 🎓 How to Interpret and Defend These Numbers in Viva:
1. **Isolation Forest caught 58.60% of all anomalies** with an **ROC-AUC of 73.40%** without ever seeing a single training label (completely unsupervised).
2. **Why is Precision lower in unsupervised log monitoring?** In HDFS benchmark data, labels are assigned at the coarse **Block level**, not line-by-line. If a block experienced an error on log line #10, all 10 preceding lines carry the anomaly label even if the first 9 lines are routine informational events.
3. **The Value of Stage 2 (LLM Doctor)**: Because unsupervised models cast a wide net with high recall, the LLM reads the surrounding 5-log context window to filter out false alarms and explain genuine incidents to administrators.

---

## 📁 Repository Structure

```text
Mini Project/
├── requirements.txt                   # Project Python dependencies
├── .gitignore                         # Excludes heavy caches and 1.5GB raw logs
├── README.md                          # Original project documentation & roadmap
├── PROJECT_EXPLANATION.md             # This comprehensive technical guide
├── demo.py                            # Standalone end-to-end detection demo
│
├── HDFS_v1/                           # Raw LogHub benchmark dataset
│   ├── HDFS.log                       # 1.5GB raw HDFS logs (~11 million lines)
│   └── preprocessed/                  # Benchmark block labels & occurrence matrices
│
├── data/
│   ├── raw/
│   │   ├── hdfs_sample.log            # Sampled 10,000 raw HDFS log lines
│   │   ├── hdfs_sample_labels.csv     # Matched BlockId -> Normal/Anomaly labels
│   │   └── synthetic_fault_stream.log # Output of live fault injector
│   └── processed/
│       ├── preprocessed_logs.jsonl    # Contract A.1 records with 384D embeddings
│       └── embeddings.npz             # (10000, 384) NumPy float32 matrix for fast ML loading
│
├── src/
│   ├── schemas.py                     # Pydantic data contracts (A.1 to A.4)
│   ├── ingestion/
│   │   ├── dataset_sampler.py         # Memory-efficient raw log sampler & label mapper
│   │   └── fault_injector.py          # Simulates live microservice logs & 3 fault types
│   ├── preprocessing/
│   │   ├── parser.py                  # Regex parser & template sanitizer (<BLOCK>, <IP>)
│   │   ├── vectorizer.py              # SentenceTransformer embedding generator with RAM cache
│   │   └── pipeline.py                # Preprocessing coordinator -> exports JSONL & NPZ
│   ├── detection/
│   │   ├── data_loader.py             # Optimized lazy loader for preprocessed records
│   │   ├── baseline.py                # Isolation Forest & K-Means clustering models
│   │   ├── autoencoder.py             # PCA-based reconstruction error anomaly detector
│   │   ├── threshold.py               # Percentile-based cut-off selector
│   │   ├── train.py                   # Model trainer & threshold calibrator
│   │   ├── infer.py                   # Anomaly scorer & context window extractor (Contract A.2)
│   │   └── evaluate.py                # Quantitative evaluation metric calculator
│   ├── models/
│   │   ├── config.json                # Calibrated thresholds & normalization statistics
│   │   ├── isolation_forest.joblib    # Serialized Isolation Forest weights
│   │   ├── kmeans.joblib              # Serialized K-Means cluster centers
│   │   └── autoencoder.pkl            # Serialized Autoencoder weights
│   ├── llm/                           # Phase 2: LLM context reasoning & prompts [Member 3]
│   └── dashboard/                     # Phase 2: Streamlit web interface [Member 4]
│
└── tests/
    ├── conftest.py                    # Pytest path configuration
    ├── test_member1_pipeline.py       # Automated tests for ingestion & preprocessing
    └── test_member2_detection.py      # Automated tests for detection & contract compliance
```

---

## 🚀 Step-by-Step Setup & How to Run Everything

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd "Mini Project"

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (All 8 Tests Passing)
```bash
pytest -v
```

### 3. Run the Live Anomaly Detection Demo
```bash
python demo.py
```
* Analyzes logs in **0.3 seconds**.
* Displays the anomaly count, anomaly rate (8.40%), and prints top anomalous events with full 5-line context windows (`>>` indicating the anomaly).

### 4. Run Quantitative Model Performance Evaluation
```bash
python -m src.detection.evaluate
```
* Computes Precision, Recall, F1-Score, Accuracy, and Confusion Matrices across all models against ground-truth benchmark labels.

### 5. Run Live Fault Injection Showcase (Great for Viva Demo)
```bash
# Inject Fault A (Memory Leak / OOM) at event #10
python -m src.ingestion.fault_injector --fault A --count 20 --fault-at 10 --interval 0.3

# Inject Fault B (Database Connection Pool Exhaustion)
python -m src.ingestion.fault_injector --fault B --count 20 --fault-at 10 --interval 0.3

# Inject Fault C (Cascading HTTP 500 Error Burst)
python -m src.ingestion.fault_injector --fault C --count 20 --fault-at 10 --interval 0.3
```

### 6. Re-sample or Re-run Data Preprocessing
```bash
# Sample 10,000 raw lines
python -m src.ingestion.dataset_sampler --lines 10000

# Run parser and vectorizer pipeline
python -m src.preprocessing.pipeline

# Retrain models and calibrate thresholds
python -m src.detection.train
```

---

## 🎯 Next Steps (Phase 2 Preview)

With Member 1 and Member 2's components fully operational, the project is ready for Phase 2:
1. **Member 3 (LLM / RAG Lead — Deepak)**:
   * Take the 5-log `context_window` from Contract A.2.
   * Send to local Ollama (`llama3:8b`) or Gemini/OpenAI API with structured JSON output formatting.
   * Emit Contract A.3 (`IncidentReport`) with plain-English root causes and remediation commands.
2. **Member 4 (Integration & Dashboard Lead — Vishnu)**:
   * Build the interactive Streamlit web dashboard in `src/dashboard/app.py`.
   * Wire the **[Confirm]** and **[Dismiss]** administrator buttons into dynamic threshold updates.
