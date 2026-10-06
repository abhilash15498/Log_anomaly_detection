"""
Synthetic Fault Injector & Metric Sampler.
Part of Member 1 (Data & Preprocessing) deliverables.

Generates realistic background server logs and simulates three distinct fault scenarios:
  1. Fault A (Memory Leak / OOM): Java heap exhaustion, GC thrashing, OutOfMemoryError.
  2. Fault B (DB Pool Exhaustion): Connection checkout timeouts, queue backpressure.
  3. Fault C (Cascading HTTP 500): Downstream service degradation, thread pool depletion.

Includes continuous psutil metrics sampling (CPU%, Memory%).
"""

import argparse
import datetime
import random
import sys
import time
from pathlib import Path
from typing import Dict, Generator, Optional
import psutil


NORMAL_MESSAGES = [
    ("INFO", "web-gateway", "GET /api/v1/products status=200 duration=24ms"),
    ("INFO", "web-gateway", "GET /api/v1/user/profile status=200 duration=18ms"),
    ("INFO", "auth-service", "Token validated successfully for user_id={user_id}"),
    ("INFO", "order-service", "Order order_{order_id} created, dispatching payment event"),
    ("INFO", "inventory-service", "Inventory reservation confirmed for item_{item_id}"),
    ("INFO", "datanode-3", "Receiving block blk_{block_id} src: /10.0.1.12:4321 dest: /10.0.1.20:50010"),
    ("INFO", "datanode-3", "Received block blk_{block_id} of size 67108864 successfully"),
    ("INFO", "payment-service", "Payment processed via gateway provider status=SUCCESS"),
]

FAULT_TEMPLATES = {
    "A": {
        "name": "Memory Leak / Out of Memory",
        "logs": [
            ("WARN", "order-service", "JVM Old Gen memory usage crossed 88% threshold (Heap: 3.8GB / 4.0GB)"),
            ("WARN", "order-service", "Full GC invocation took 3120ms; reclaimed only 12MB"),
            ("ERROR", "order-service", "java.lang.OutOfMemoryError: Java heap space at com.app.order.OrderBuffer.allocate(OrderBuffer.java:184)"),
            ("CRITICAL", "order-service", "Worker thread thread-pool-exec-4 died abruptly due to unhandled OutOfMemoryError"),
            ("ERROR", "web-gateway", "Upstream server order-service:8080 unreachable (Connection reset by peer)"),
        ],
        "metric_modifier": {"cpu_delta": 40.0, "memory_delta": 35.0}
    },
    "B": {
        "name": "Database Connection Pool Exhaustion",
        "logs": [
            ("WARN", "inventory-service", "HikariPool-1 - Connection acquisition time exceeded 5000ms threshold"),
            ("WARN", "inventory-service", "HikariPool-1 - Active connections: 50/50, Pending requests: 142"),
            ("ERROR", "inventory-service", "org.postgresql.util.PSQLException: Connection checkout timed out after 30000ms"),
            ("ERROR", "inventory-service", "Failed to acquire database lock for inventory stock check item_{item_id}"),
            ("ERROR", "web-gateway", "HTTP 504 Gateway Timeout on POST /api/v1/cart/checkout from inventory-service"),
        ],
        "metric_modifier": {"cpu_delta": 25.0, "memory_delta": 10.0}
    },
    "C": {
        "name": "Cascading HTTP 500 Error Burst",
        "logs": [
            ("WARN", "auth-service", "Rate limiter storage Redis connection dropped, falling back to local memory"),
            ("ERROR", "auth-service", "NullPointerException in TokenSignatureVerifier.verifyKey()"),
            ("ERROR", "web-gateway", "HTTP 500 Internal Server Error returned for GET /api/v1/auth/verify"),
            ("ERROR", "web-gateway", "Cascading failure: CircuitBreaker 'auth-service' opened (error rate 78.4%)"),
            ("CRITICAL", "web-gateway", "Drop rate 92% of incoming requests; client retry storms detected"),
        ],
        "metric_modifier": {"cpu_delta": 65.0, "memory_delta": 15.0}
    }
}


def get_current_metrics(cpu_delta: float = 0.0, memory_delta: float = 0.0) -> Dict[str, float]:
    """Capture system metrics via psutil with optional simulated fault delta."""
    try:
        raw_cpu = psutil.cpu_percent(interval=None)
        raw_mem = psutil.virtual_memory().percent
    except Exception:
        raw_cpu = 15.0
        raw_mem = 45.0

    # Clamp percentages between 0.0 and 100.0
    sim_cpu = min(100.0, max(0.0, raw_cpu + cpu_delta))
    sim_mem = min(100.0, max(0.0, raw_mem + memory_delta))

    return {
        "cpu_percent": round(sim_cpu, 2),
        "memory_percent": round(sim_mem, 2)
    }


def format_log_line(timestamp_str: str, level: str, source: str, message: str) -> str:
    """Standard raw text log format."""
    return f"{timestamp_str} [{level}] [{source}]: {message}"


def generate_log_event(
    fault_type: Optional[str] = None,
    fault_index: int = 0
) -> Dict:
    """Generates a single log event dict with text and metrics."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if fault_type and fault_type in FAULT_TEMPLATES:
        fault_info = FAULT_TEMPLATES[fault_type]
        logs = fault_info["logs"]
        idx = fault_index % len(logs)
        level, source, template = logs[idx]
        msg = template.format(
            item_id=random.randint(1000, 9999),
            user_id=random.randint(100, 999),
            order_id=random.randint(50000, 99999)
        )
        deltas = fault_info["metric_modifier"]
        metrics = get_current_metrics(cpu_delta=deltas["cpu_delta"], memory_delta=deltas["memory_delta"])
        is_fault = True
        fault_name = fault_info["name"]
    else:
        level, source, template = random.choice(NORMAL_MESSAGES)
        msg = template.format(
            user_id=random.randint(100, 999),
            order_id=random.randint(50000, 99999),
            item_id=random.randint(1000, 9999),
            block_id=random.randint(100000000, 999999999)
        )
        metrics = get_current_metrics()
        is_fault = False
        fault_name = None

    raw_text = format_log_line(now, level, source, msg)
    return {
        "timestamp": now,
        "level": level,
        "source": source,
        "message": msg,
        "raw_text": raw_text,
        "metrics": metrics,
        "is_fault": is_fault,
        "fault_type": fault_type,
        "fault_name": fault_name
    }


def stream_logs(
    output_file: Optional[str] = None,
    interval_sec: float = 1.0,
    total_events: int = 50,
    inject_fault_at: int = 20,
    fault_type: str = "A"
):
    """Streams logs sequentially to stdout and optionally appends to output_file."""
    out_path = Path(output_file) if output_file else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[FaultInjector] Starting stream: {total_events} events. Fault '{fault_type}' scheduled at event #{inject_fault_at}...")

    fault_step = 0
    in_fault = False

    for i in range(1, total_events + 1):
        if i == inject_fault_at:
            in_fault = True
            print(f"\n🚨 [INJECTING FAULT {fault_type}: {FAULT_TEMPLATES[fault_type]['name']}] 🚨")

        if in_fault:
            event = generate_log_event(fault_type=fault_type, fault_index=fault_step)
            fault_step += 1
            if fault_step >= len(FAULT_TEMPLATES[fault_type]["logs"]):
                in_fault = False  # End of burst
        else:
            event = generate_log_event()

        log_line = event["raw_text"]
        print(f"[{i:02d}/{total_events:02d}] {log_line} (CPU: {event['metrics']['cpu_percent']}%, RAM: {event['metrics']['memory_percent']}%)")

        if out_path:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

        if interval_sec > 0 and i < total_events:
            time.sleep(interval_sec)

    print("[FaultInjector] Finished stream generation.")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Fault Injector for AI Infrastructure Monitoring")
    parser.add_argument("--fault", type=str, choices=["A", "B", "C"], default="A", help="Fault scenario (A: OOM, B: DB Pool, C: HTTP 500)")
    parser.add_argument("--out", type=str, default="data/raw/synthetic_fault_stream.log", help="Path to write generated logs")
    parser.add_argument("--count", type=int, default=30, help="Total number of events to generate")
    parser.add_argument("--fault-at", type=int, default=15, help="Step index at which to trigger the fault burst")
    parser.add_argument("--interval", type=float, default=0.2, help="Interval in seconds between logs (0 for instant batch)")
    args = parser.parse_args()

    stream_logs(
        output_file=args.out,
        interval_sec=args.interval,
        total_events=args.count,
        inject_fault_at=args.fault_at,
        fault_type=args.fault
    )


if __name__ == "__main__":
    main()
