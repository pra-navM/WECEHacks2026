"""Realistic Shortest Job First (SJF) scheduling policy."""

from __future__ import annotations
from typing import List, Optional
from simulator.scheduler_base import SchedulerBase
from simulator.task import Task

class SJFScheduler(SchedulerBase):
    """
    Non-preemptive SJF with Estimated Burst Time.
    This version is 'fair' because it doesn't know the future; 
    it guesses based on task metadata.
    """

    def __init__(self) -> None:
        self._queue: List[Task] = []

    def _estimate_burst(self, task: Task) -> float:
        """
        A 'fair' heuristic: Real OSs often use memory usage (RSS) 
        or I/O probability to guess task length.
        """
        # Heuristic: Assume high memory tasks are 'elephants' (longer)
        # and high I/O tasks are 'mice' (shorter).
        base_guess = 50 
        memory_bias = task.metadata.memory_rss * 0.5
        io_bias = task.metadata.io_blocking_prob * -20
        return base_guess + memory_bias + io_bias

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int, **kwargs: object) -> Optional[Task]:
        if not self._queue:
            return None
        
        # SORT BY ESTIMATE, not ground truth remaining_time
        self._queue.sort(
            key=lambda t: (self._estimate_burst(t), t.arrival_time, t.task_id)
        )
        return self._queue.pop(0)

    def on_task_complete(self, task: Task) -> None:
        pass

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0