"""First Come First Served (FCFS) scheduling policy."""

from __future__ import annotations

from collections import deque
from typing import Optional

from simulator.scheduler_base import SchedulerBase
from simulator.task import Task


class FCFSScheduler(SchedulerBase):
    """Non-preemptive FIFO scheduler.

    Tasks are dispatched in the order they were added to the ready queue.
    """

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int, **kwargs: object) -> Optional[Task]:
        if self._queue:
            return self._queue.popleft()
        return None

    def on_task_complete(self, task: Task) -> None:
        pass

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0
