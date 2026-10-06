"""
End-to-End Preprocessing Pipeline.
Part of Member 1 (Data & Preprocessing) deliverables.

Coordinates reading raw logs, parsing them into structured templates,
generating semantic embeddings, and persisting Schema Contract A.1 records
for Member 2's anomaly detection models.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.preprocessing.parser import parse_log_line
from src.preprocessing.vectorizer import LogVectorizer
from src.schemas import PreprocessedLogRecord


def run_preprocessing_pipeline(
    raw_log_path: str = "data/raw/hdfs_sample.log",
    label_path: Optional[str] = "data/raw/hdfs_sample_labels.csv",
    output_jsonl_path: str = "data/processed/preprocessed_logs.jsonl",
    output_npz_path: str = "data/processed/embeddings.npz",
    max_records: Optional[int] = None,
    use_sentence_transformers: bool = True,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, int]:
    """
    Executes the ingestion -> parsing -> vectorization -> export pipeline.
    """
    raw_path = Path(raw_log_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw log file not found at: {raw_path.resolve()}")

    out_jsonl = Path(output_jsonl_path)
    out_npz = Path(output_npz_path)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # Load ground truth labels map if available (BlockId -> Label)
    label_map: Dict[str, str] = {}
    if label_path and Path(label_path).exists():
        print(f"[Pipeline] Loading ground-truth labels from {label_path}...")
        df_lbl = pd.read_csv(label_path)
        label_map = dict(zip(df_lbl["BlockId"], df_lbl["Label"]))

    # Step 1: Read and parse log lines
    print(f"[Pipeline] Reading and parsing raw logs from {raw_path}...")
    parsed_records: List[Dict] = []
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if max_records and idx >= max_records:
                break
            if not line.strip():
                continue
            parsed = parse_log_line(line, line_idx=idx)
            parsed_records.append(parsed)

    total_parsed = len(parsed_records)
    print(f"[Pipeline] Parsed {total_parsed:,} log lines.")

    # Step 2: Initialize vectorizer
    vectorizer = LogVectorizer(
        model_name=model_name,
        use_sentence_transformers=use_sentence_transformers
    )

    # Step 3: Vectorize templates in batches
    templates = [r["template"] for r in parsed_records]
    print(f"[Pipeline] Computing embeddings for {len(templates):,} logs (unique templates: {len(set(templates)):,})...")
    embeddings = vectorizer.encode_batch(templates, batch_size=256)

    # Step 4: Write Contract A.1 JSONL and build evaluation arrays
    print(f"[Pipeline] Exporting Contract A.1 records to {out_jsonl}...")
    log_ids: List[str] = []
    ground_truth_labels: List[int] = []  # 0 for Normal, 1 for Anomaly, -1 for Unknown
    block_ids: List[str] = []

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for r, emb in zip(parsed_records, embeddings):
            record = PreprocessedLogRecord(
                log_id=r["log_id"],
                timestamp=r["timestamp"],
                raw_text=r["raw_text"],
                source=r["source"],
                embedding=emb,
                metrics=None
            )
            f_out.write(record.model_dump_json() + "\n")

            log_ids.append(r["log_id"])
            blk = r.get("block_id") or ""
            block_ids.append(blk)

            lbl_str = label_map.get(blk, "Unknown")
            if lbl_str == "Anomaly":
                ground_truth_labels.append(1)
            elif lbl_str == "Normal":
                ground_truth_labels.append(0)
            else:
                ground_truth_labels.append(-1)

    # Step 5: Save compressed numpy package for fast loading by Member 2
    embeddings_mat = np.array(embeddings, dtype=np.float32)
    labels_arr = np.array(ground_truth_labels, dtype=np.int32)
    np.savez_compressed(
        out_npz,
        embeddings=embeddings_mat,
        labels=labels_arr,
        log_ids=np.array(log_ids),
        block_ids=np.array(block_ids)
    )
    print(f"[Pipeline] Saved fast matrix file to {out_npz} (Shape: {embeddings_mat.shape})")

    normal_count = sum(1 for l in ground_truth_labels if l == 0)
    anomaly_count = sum(1 for l in ground_truth_labels if l == 1)
    unknown_count = sum(1 for l in ground_truth_labels if l == -1)

    summary = {
        "total_records": total_parsed,
        "embedding_dim": embeddings_mat.shape[1],
        "normal_records": normal_count,
        "anomaly_records": anomaly_count,
        "unknown_records": unknown_count
    }
    print(f"[Pipeline] Preprocessing complete! Summary:\n{json.dumps(summary, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Preprocessing Pipeline (Member 1)")
    parser.add_argument("--raw", type=str, default="data/raw/hdfs_sample.log", help="Path to raw logs")
    parser.add_argument("--labels", type=str, default="data/raw/hdfs_sample_labels.csv", help="Path to ground truth labels CSV")
    parser.add_argument("--out-jsonl", type=str, default="data/processed/preprocessed_logs.jsonl", help="Output JSONL path")
    parser.add_argument("--out-npz", type=str, default="data/processed/embeddings.npz", help="Output NPZ path")
    parser.add_argument("--max", type=int, default=None, help="Maximum records to process")
    parser.add_argument("--offline", action="store_true", help="Use fast offline hashing instead of neural transformer")
    args = parser.parse_args()

    run_preprocessing_pipeline(
        raw_log_path=args.raw,
        label_path=args.labels,
        output_jsonl_path=args.out_jsonl,
        output_npz_path=args.out_npz,
        max_records=args.max,
        use_sentence_transformers=not args.offline
    )


if __name__ == "__main__":
    main()
