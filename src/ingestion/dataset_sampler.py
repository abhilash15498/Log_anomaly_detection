"""
Dataset Sampler for HDFS LogHub Dataset.
Part of Member 1 (Data & Preprocessing) deliverables.

Extracts a representative, manageable sample from large raw logs (e.g. HDFS.log)
and pairs log blocks with ground-truth labels for evaluation.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


BLOCK_REGEX = re.compile(r'(blk_[-0-9]+)')


def load_ground_truth_labels(label_file_path: Path) -> Dict[str, str]:
    """Load block_id -> label (Normal/Anomaly) dictionary if available."""
    if not label_file_path.exists():
        print(f"[Sampler] Warning: Label file not found at {label_file_path}. Labels will be omitted.")
        return {}
    
    print(f"[Sampler] Loading labels from {label_file_path}...")
    df = pd.read_csv(label_file_path)
    return dict(zip(df["BlockId"], df["Label"]))


def sample_hdfs_logs(
    source_log_path: str = "HDFS_v1/HDFS.log",
    label_path: str = "HDFS_v1/preprocessed/anomaly_label.csv",
    output_log_path: str = "data/raw/hdfs_sample.log",
    output_labels_path: str = "data/raw/hdfs_sample_labels.csv",
    num_lines: int = 25000,
) -> int:
    """
    Extracts the first `num_lines` from the large raw HDFS log into data/raw/
    and creates a corresponding sample ground-truth label map.
    """
    src_path = Path(source_log_path)
    lbl_path = Path(label_path)
    out_log = Path(output_log_path)
    out_lbl = Path(output_labels_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Source raw log file not found at {src_path.resolve()}")

    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    block_labels = load_ground_truth_labels(lbl_path)
    sampled_block_labels: Dict[str, str] = {}
    
    count = 0
    normal_count = 0
    anomaly_count = 0
    unknown_count = 0

    print(f"[Sampler] Sampling {num_lines:,} lines from {src_path} -> {out_log}...")
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(out_log, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if count >= num_lines:
                break
            f_out.write(line)
            count += 1

            # Extract block ID and check label
            match = BLOCK_REGEX.search(line)
            if match:
                block_id = match.group(1)
                label = block_labels.get(block_id, "Unknown")
                sampled_block_labels[block_id] = label
                if label == "Anomaly":
                    anomaly_count += 1
                elif label == "Normal":
                    normal_count += 1
                else:
                    unknown_count += 1

    # Save sampled block labels for Member 2's evaluation
    if sampled_block_labels:
        df_sample_labels = pd.DataFrame(
            list(sampled_block_labels.items()),
            columns=["BlockId", "Label"]
        )
        df_sample_labels.to_csv(out_lbl, index=False)
        print(f"[Sampler] Saved {len(sampled_block_labels):,} block labels to {out_lbl}")

    print(
        f"[Sampler] Done! Sampled {count:,} log lines.\n"
        f"          Lines with Normal blocks: {normal_count:,}\n"
        f"          Lines with Anomaly blocks: {anomaly_count:,}\n"
        f"          Unique Blocks in sample: {len(sampled_block_labels):,}"
    )
    return count


def main():
    parser = argparse.ArgumentParser(description="Sample raw HDFS logs for development & Phase 1 review")
    parser.add_argument("--source", type=str, default="HDFS_v1/HDFS.log", help="Path to raw HDFS.log")
    parser.add_argument("--labels", type=str, default="HDFS_v1/preprocessed/anomaly_label.csv", help="Path to anomaly labels CSV")
    parser.add_argument("--out-log", type=str, default="data/raw/hdfs_sample.log", help="Output sampled log path")
    parser.add_argument("--out-labels", type=str, default="data/raw/hdfs_sample_labels.csv", help="Output sampled labels path")
    parser.add_argument("--lines", type=int, default=25000, help="Number of lines to sample")
    args = parser.parse_args()

    sample_hdfs_logs(
        source_log_path=args.source,
        label_path=args.labels,
        output_log_path=args.out_log,
        output_labels_path=args.out_labels,
        num_lines=args.lines
    )


if __name__ == "__main__":
    main()
