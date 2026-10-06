# Team Coordination & Step-by-Step Build Guide

> **Project**: AI-Driven Infrastructure Monitoring and Intelligent Incident Recommendation System  
> **Course**: BAI506 Mini Project — Academic Year 2026–27  
> **Institution**: Dept. of AI & ML, BMSIT&M | **Guide**: Dr. Archana Bhat  
> **Batch**: C3  

---

## 👥 The Team & Roles

| Member | Role | USN | Core Ownership |
|---|---|---|---|
| **Member 1** | **Data & Preprocessing Lead** | `1BY24AI002` (Abhilash Hiremath) | Log dataset acquisition, live fault injection, Drain3 parsing, embedding vectors |
| **Member 2** | **ML Detection Lead** |  `1BY24AI157` (Shrikrishna R Prabhu) | Isolation Forest baseline, PyTorch Autoencoder, anomaly scoring & evaluation |
| **Member 3** | **LLM / RAG Lead** |`1BY24AI037` (Deepak Suresh Naik) | Context aggregation, prompt engineering, LLM integration (Ollama/API), RAG |
| **Member 4** | **Integration & Dashboard Lead** | `1BY24AI191` (Vishnu Karanth A) | Streamlit dashboard, feedback loop wiring, end-to-end integration tests |

---

## 💡 The Core Mental Model (How the System Works)

Imagine you are building a **smart security scout and expert doctor** for computer servers:
* **The Scout (Lightweight ML Detector)**: Continuously inspects millions of incoming log lines in milliseconds. 99.9% of normal logs stop here without bothering anyone.
* **The Doctor (LLM Reasoning Layer)**: Wakes up **only** when the Scout raises a red flag. It reads the surrounding log window, diagnoses the root cause in plain English, and provides concrete remediation commands.
* **The Administrator Dashboard (Streamlit)**: Surfaces the alerts and diagnoses to the engineer. The engineer can **Confirm** or **Dismiss** (false alarm). Dismissals feed back into the Scout to automatically adjust sensitivity thresholds and curb alert fatigue.

---

## 🗺️ Step-by-Step Implementation Roadmap

```text
+-----------------------+     +--------------------------+     +------------------------+
| Step 1: Raw Logs      | --> | Step 2: Vectorization    | --> | Step 3: ML Detector    |
| (Ingest & Faults)     |     | (sentence-transformers)  |     | (Isolation Forest/AE)  |
| Owned by: Member 1    |     | Owned by: Member 1       |     | Owned by: Member 2     |
+-----------------------+     +--------------------------+     +------------------------+
                                                                            |
                                                                   (Only if Anomaly!)
                                                                            v
+-----------------------+     +--------------------------+     +------------------------+
| Step 6: Full Demo     | <-- | Step 5: Web Dashboard    | <-- | Step 4: LLM Diagnosis |
| (Fault Injection Test)|     | (Streamlit UI + Feedback)|     | (Context Window + Prompt)
| Owned by: All (M4 lead|     | Owned by: Member 4       |     | Owned by: Member 3     |
+-----------------------+     +--------------------------+     +------------------------+
```

---

### Step 1: Dataset Acquisition & Fault-Injection Setup
* **Lead**: Member 1
* **Objective**: Prepare raw logs and create a live demo environment.
* **Tasks**:
  1. Download a subset of **LogHub HDFS_v1** (or BGL) logs and store raw samples in `data/raw/`.
  2. Write a Python script/service that emits normal server logs every 1–2 seconds.
  3. Implement a **fault injection script** with at least 3 distinct failure scenarios:
     * **Fault A**: Memory leak / Out-of-Memory (`java.lang.OutOfMemoryError`).
     * **Fault B**: Database connection timeout / pool exhaustion.
     * **Fault C**: Cascading HTTP 500 error / unauthorized spike.
* **Handoff Output**: A steady stream or file of raw text logs.

---

### Step 2: Log Parsing & Vector Embeddings
* **Lead**: Member 1
* **Objective**: Convert English text into numerical vectors that machine learning algorithms can compute.
* **Tasks**:
  1. (Optional) Parse dynamic parameters (IP addresses, IDs) using regex or Drain3.
  2. Use `sentence-transformers` (`all-MiniLM-L6-v2`) to turn each log message into a 384-dimensional list of floats.
  3. Produce records conforming strictly to **Contract A.1**:
     ```json
     {
       "log_id": "log_001",
       "timestamp": "2026-09-22T19:00:00Z",
       "raw_text": "Failed to write block blk_1049281 to datanode",
       "source": "datanode-1",
       "embedding": [0.012, -0.045, 0.089, "..."]
     }
     ```
* **Handoff Output**: Clean preprocessed dataset saved in `data/processed/` for Member 2.

---

### Step 3: Anomaly Detection Models
* **Lead**: Member 2
* **Objective**: Train unsupervised ML models to detect abnormal log patterns.
* **Tasks**:
  1. **Baseline Model**: Train an `IsolationForest` on normal log embeddings using `scikit-learn`.
  2. **Deep Model**: Implement a PyTorch `Autoencoder` (e.g., 384 $\to$ 128 $\to$ 32 $\to$ 128 $\to$ 384) trained on normal data to minimize reconstruction error.
  3. **Threshold Selection**: Calibrate the anomaly score cutoff (e.g. 99th percentile of normal scores).
  4. Produce records conforming strictly to **Contract A.2**:
     ```json
     {
       "log_id": "log_001",
       "anomaly_score": 0.88,
       "is_anomaly": true,
       "threshold_used": 0.75,
       "context_window": ["log_prev_2", "log_prev_1", "log_001", "log_next_1"]
     }
     ```
  5. Compute evaluation metrics: **Precision, Recall, F1-Score, and Accuracy** on labeled test data.
* **Handoff Output**: Trained model weights and detection output stream for Members 3 & 4.

---

### Step 4: Context Aggregation & LLM Root-Cause Reasoning
* **Lead**: Member 3
* **Objective**: Translate raw anomalous log events into plain English explanations and remediation steps.
* **Tasks**:
  1. When an anomaly is flagged, collect the preceding $N$ logs and subsequent $M$ logs using the `context_window` IDs.
  2. Design a structured prompt demanding strict JSON output.
  3. Connect to an LLM provider:
     * **Local (Free & Offline)**: Ollama (`llama3:8b` or `mistral:7b`).
     * **API-based**: Gemini API or OpenAI API.
     * *(Use a mock client during early development to avoid rate limits).*
  4. Produce output conforming strictly to **Contract A.3**:
     ```json
     {
       "log_id": "log_001",
       "explanation": "DataNode failed to write block due to storage volume exhaustion.",
       "severity": "critical",
       "recommended_actions": [
         "Run disk cleanup on /data partition",
         "Increase DFS replication timeout",
         "Restart DataNode daemon"
       ],
       "retrieved_context": []
     }
     ```
* **Handoff Output**: Incident reports ready to be rendered in Member 4's dashboard.

---

### Step 5: Administrator Dashboard & Feedback Loop
* **Lead**: Member 4
* **Objective**: Create the user interface for administrators to monitor incidents and tune the system.
* **Tasks**:
  1. Build a **Streamlit** dashboard in `src/dashboard/app.py`:
     * **Live Log Feed**: Scrolling list of recent logs (normal vs. red-highlighted anomalies).
     * **Incident Inspection Panel**: Displays the LLM's explanation, severity badge, and remediation actions.
     * **Feedback Controls**: Two buttons for each alert:
       * ✅ **Confirm Anomaly**
       * ❌ **Dismiss (False Positive)**
  2. Store feedback in **Contract A.4**:
     ```json
     {
       "log_id": "log_001",
       "administrator_verdict": "dismissed",
       "timestamp": "2026-09-22T19:05:00Z"
     }
     ```
  3. Feedback loop: Nudge the detection threshold up when dismissals occur, reducing false alerts over time.
* **Handoff Output**: An interactive, running web application.

---

### Step 6: End-to-End Integration & Viva Rehearsal
* **Lead**: All Members (Coordinated by Member 4)
* **Objective**: Wire all four components together and demonstrate live fault injection.
* **Tasks**:
  1. Run end-to-end integration test (`tests/test_pipeline.py`) validating the data flow:
     $$\text{Ingestion} \longrightarrow \text{Detection} \longrightarrow \text{LLM Reasoning} \longrightarrow \text{Dashboard}$$
  2. Conduct a **Live Fault Injection Demo**:
     * Start the system running smoothly on normal logs.
     * Trigger Fault 1 (Memory Leak).
     * Show the detection score spike on the dashboard.
     * Show the LLM diagnostic explanation appear in real time.
     * Click "Confirm" or "Dismiss" to showcase adaptive thresholding.
  3. Prepare for the viva: Every member practices explaining their stage and how it connects to the rest of the pipeline.

---

## 🤝 Weekly Team Sync Checklist

Use this 15-minute weekly checklist during team meetings:
- [ ] **Data Contract Check**: Did anyone change any field names in `src/schemas.py`? *(Any change must be agreed by all four).*
- [ ] **Independent Testing**: Does your module run on dummy input data without needing the other members' code?
- [ ] **Cross-Explanation Practice**: Can Member 1 explain how the Autoencoder works? Can Member 2 explain the LLM prompt? Can Member 3 explain the Streamlit feedback? Can Member 4 explain the embeddings?
