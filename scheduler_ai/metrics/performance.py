"""Performance metrics collection and reporting.

Phase 2 will implement detailed metric aggregation and comparison
across scheduling policies (throughput, fairness, utilization, etc.).
"""

from __future__ import annotations

from typing import Dict, List

from scheduler_ai.simulator.task import Task


def compute_metrics(tasks: List[Task]) -> Dict[str, float]:
    """Compute summary statistics for a completed simulation run.

    Args:
        tasks: Completed task list from Simulation.run().

    Returns:
        Dictionary of metric name to value.
    """
    raise NotImplementedError("Phase 2: metrics collection")
