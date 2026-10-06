"""
Log Parser & Sanitizer.
Part of Member 1 (Data & Preprocessing) deliverables.

Extracts structured metadata (timestamp, level, component/source, message)
from raw logs (HDFS format or synthetic standard format) and sanitizes dynamic tokens.
"""

import re
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple


# Regex pattern for LogHub HDFS log format:
# Example: "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
HDFS_REGEX = re.compile(
    r'^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>[A-Z]+)\s+(?P<component>[\w\.\$\_]+):\s+(?P<message>.*)$'
)

# Regex pattern for standard synthetic/server log format:
# Example: "2026-10-06T08:50:00.123Z [ERROR] [order-service]: OutOfMemoryError..."
STANDARD_REGEX = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z)?)\s+\[(?P<level>[A-Z]+)\]\s+\[(?P<source>[\w\.\-]+)\]:\s+(?P<message>.*)$'
)

# Sanitization regular expressions
BLOCK_REGEX = re.compile(r'blk_[-0-9]+')
IP_REGEX = re.compile(r'/(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?')
HEX_REGEX = re.compile(r'0x[0-9a-fA-F]+')
NUMBER_REGEX = re.compile(r'\b\d+\b')


def parse_hdfs_timestamp(date_str: str, time_str: str) -> str:
    """Converts HDFS date (YYMMDD) and time (HHMMSS) into ISO 8601 string."""
    try:
        dt = datetime.strptime(f"20{date_str}{time_str}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def sanitize_message(raw_msg: str) -> str:
    """
    Sanitizes dynamic variables (Block IDs, IPs, numbers) from log text
    to reveal the underlying semantic pattern/template.
    """
    cleaned = BLOCK_REGEX.sub("<BLOCK>", raw_msg)
    cleaned = IP_REGEX.sub("<IP>", cleaned)
    cleaned = HEX_REGEX.sub("<HEX>", cleaned)
    cleaned = NUMBER_REGEX.sub("<NUM>", cleaned)
    return cleaned.strip()


def extract_block_id(raw_msg: str) -> Optional[str]:
    """Extracts HDFS block ID if present."""
    match = BLOCK_REGEX.search(raw_msg)
    return match.group(0) if match else None


def parse_log_line(raw_line: str, line_idx: int = 0) -> Dict:
    """
    Parses a single log line into structured components.
    Supports both HDFS logs and standard microservice server logs.
    """
    raw_line_clean = raw_line.strip()
    log_id = f"log_{line_idx:07d}"

    # Try HDFS regex first
    m_hdfs = HDFS_REGEX.match(raw_line_clean)
    if m_hdfs:
        d = m_hdfs.groupdict()
        timestamp = parse_hdfs_timestamp(d["date"], d["time"])
        level = d["level"]
        source = d["component"].replace(":", "")
        message = d["message"]
        block_id = extract_block_id(message)
        template = sanitize_message(message)
        return {
            "log_id": log_id,
            "timestamp": timestamp,
            "level": level,
            "source": source,
            "raw_text": raw_line_clean,
            "message": message,
            "template": template,
            "block_id": block_id,
            "format": "hdfs"
        }

    # Try Standard microservice regex
    m_std = STANDARD_REGEX.match(raw_line_clean)
    if m_std:
        d = m_std.groupdict()
        timestamp = d["timestamp"]
        level = d["level"]
        source = d["source"]
        message = d["message"]
        template = sanitize_message(message)
        return {
            "log_id": log_id,
            "timestamp": timestamp,
            "level": level,
            "source": source,
            "raw_text": raw_line_clean,
            "message": message,
            "template": template,
            "block_id": extract_block_id(message),
            "format": "standard"
        }

    # Fallback for unformatted lines
    return {
        "log_id": log_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "source": "unknown",
        "raw_text": raw_line_clean,
        "message": raw_line_clean,
        "template": sanitize_message(raw_line_clean),
        "block_id": extract_block_id(raw_line_clean),
        "format": "raw"
    }
