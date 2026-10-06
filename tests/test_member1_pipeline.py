"""
Unit tests for Member 1 (Data & Preprocessing) pipeline.
Tests parsing, vectorization, fault generation, and Schema Contract A.1 adherence.
"""

import os
from pathlib import Path
import pytest

from src.ingestion.fault_injector import generate_log_event, FAULT_TEMPLATES
from src.preprocessing.parser import parse_log_line, sanitize_message
from src.preprocessing.vectorizer import LogVectorizer
from src.schemas import PreprocessedLogRecord


def test_hdfs_log_parsing():
    raw_hdfs = (
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
        "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    )
    parsed = parse_log_line(raw_hdfs, line_idx=1)
    assert parsed["level"] == "INFO"
    assert "DataNode$DataXceiver" in parsed["source"]
    assert parsed["block_id"] == "blk_-1608999687919862906"
    assert "<BLOCK>" in parsed["template"]
    assert "<IP>" in parsed["template"]


def test_standard_log_parsing():
    raw_std = "2026-10-06T14:00:00Z [ERROR] [order-service]: java.lang.OutOfMemoryError: Java heap space"
    parsed = parse_log_line(raw_std, line_idx=2)
    assert parsed["level"] == "ERROR"
    assert parsed["source"] == "order-service"
    assert "OutOfMemoryError" in parsed["message"]


def test_fault_injector_scenarios():
    for f_type in ["A", "B", "C"]:
        event = generate_log_event(fault_type=f_type, fault_index=0)
        assert event["is_fault"] is True
        assert event["fault_type"] == f_type
        assert "cpu_percent" in event["metrics"]
        assert "memory_percent" in event["metrics"]
        assert 0.0 <= event["metrics"]["cpu_percent"] <= 100.0
        assert 0.0 <= event["metrics"]["memory_percent"] <= 100.0


def test_vectorizer_contract_compliance():
    vectorizer = LogVectorizer(use_sentence_transformers=False, vector_dim=64)
    record = vectorizer.create_preprocessed_record(
        log_id="log_0000001",
        timestamp="2026-10-06T14:00:00Z",
        raw_text="Verification succeeded for blk_123",
        source="datanode-1",
        text_to_embed="Verification succeeded for <BLOCK>",
        metrics={"cpu_percent": 12.5, "memory_percent": 45.0}
    )

    assert isinstance(record, PreprocessedLogRecord)
    assert record.log_id == "log_0000001"
    assert len(record.embedding) == 64
    assert record.metrics["cpu_percent"] == 12.5

    # Check Pydantic JSON serialization roundtrip
    json_str = record.model_dump_json()
    reconstructed = PreprocessedLogRecord.model_validate_json(json_str)
    assert reconstructed.log_id == record.log_id
    assert reconstructed.embedding == record.embedding


def test_processed_dataset_exists_and_valid():
    npz_path = Path("data/processed/embeddings.npz")
    jsonl_path = Path("data/processed/preprocessed_logs.jsonl")

    assert npz_path.exists(), "embeddings.npz was not generated"
    assert jsonl_path.exists(), "preprocessed_logs.jsonl was not generated"
    assert npz_path.stat().st_size > 0
    assert jsonl_path.stat().st_size > 0
