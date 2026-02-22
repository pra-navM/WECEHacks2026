"""Performance metrics collection and reporting.

Phase 2 will implement detailed metric aggregation and comparison
across scheduling policies (throughput, fairness, utilization, etc.).
"""

from __future__ import annotations

from typing import Dict, List

from simulator.task import Task
import numpy as np


def compute_metrics(tasks: List[Task]) -> Dict[str, float]:
    if not tasks:
        return {
            "avg_turnaround": 0.0,
            "avg_waiting_time": 0.0,
            "max_turnaround": 0.0,
            "p99_latency": 0.0, # Add placeholder
            "throughput": 0.0,
        }

    turnarounds = [t.turnaround_time for t in tasks if t.turnaround_time is not None]
    wait_times = [t.waiting_time for t in tasks if t.waiting_time is not None]
    completion_times = [t.completion_time for t in tasks if t.completion_time is not None]

    n = len(tasks)
    avg_turnaround = sum(turnarounds) / n if turnarounds else 0.0
    avg_waiting_time = sum(wait_times) / n if wait_times else 0.0
    max_turnaround = max(turnarounds) if turnarounds else 0.0
    
    # Calculate P99 Tail Latency
    p99_latency = np.percentile(wait_times, 99) if wait_times else 0.0

    time_span = max(1, max(completion_times)) if completion_times else 1
    throughput = (100.0 * n) / time_span

    return {
        "avg_turnaround": avg_turnaround,
        "avg_waiting_time": avg_waiting_time,
        "max_turnaround": max_turnaround,
        "p99_latency": p99_latency, # Return P99
        "throughput": throughput,
    }