"""Shortest Job First (SJF) scheduling policy."""

from __future__ import annotations

from typing import List, Optional

from simulator.scheduler_base import SchedulerBase
from simulator.task import Task


class SJFScheduler(SchedulerBase):
    """Non-preemptive Shortest Job First scheduler.

    When a core becomes idle, the task with the smallest remaining_time
    is selected. Ties are broken by arrival_time, then task_id for
    determinism.
    """

    def __init__(self) -> None:
        self._queue: List[Task] = []

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int) -> Optional[Task]:
        if not self._queue:
            return None
        self._queue.sort(
            key=lambda t: (t.remaining_time, t.arrival_time, t.task_id)
        )
        return self._queue.pop(0)

    def on_task_complete(self, task: Task) -> None:
        pass

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0
