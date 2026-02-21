"""Performance metrics collection and reporting.

Phase 2 will implement detailed metric aggregation and comparison
across scheduling policies (throughput, fairness, utilization, etc.).
"""

from __future__ import annotations

from typing import Dict, List

from simulator.task import Task


def compute_metrics(tasks: List[Task]) -> Dict[str, float]:
    """Compute summary statistics for a completed simulation run.

    Args:
        tasks: Completed task list from Simulation.run().

    Returns:
        Dictionary of metric name to value.
    """
    if not tasks:
        return {
            "avg_turnaround": 0.0,
            "avg_waiting_time": 0.0,
            "max_turnaround": 0.0,
            "throughput": 0.0,
        }

    turnarounds = []
    wait_times = []
    completion_times = []

    for t in tasks:
        if t.turnaround_time is not None:
            turnarounds.append(t.turnaround_time)
        if t.waiting_time is not None:
            wait_times.append(t.waiting_time)
        if t.completion_time is not None:
            completion_times.append(t.completion_time)

    n = len(tasks)
    avg_turnaround = sum(turnarounds) / n if turnarounds else 0.0
    avg_waiting_time = sum(wait_times) / n if wait_times else 0.0
    max_turnaround = max(turnarounds) if turnarounds else 0.0

    time_span = 1
    if completion_times:
        time_span = max(1, max(completion_times))
    throughput = (100.0 * n) / time_span if time_span else 0.0

    return {
        "avg_turnaround": avg_turnaround,
        "avg_waiting_time": avg_waiting_time,
        "max_turnaround": max_turnaround,
        "throughput": throughput,
    }
