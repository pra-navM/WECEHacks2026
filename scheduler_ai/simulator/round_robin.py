"""Round Robin scheduling policy with time-quantum preemption."""

from __future__ import annotations

from collections import deque
from typing import Optional

from scheduler_ai.simulator.scheduler_base import SchedulerBase
from scheduler_ai.simulator.task import Task


class RoundRobinScheduler(SchedulerBase):
    """Preemptive Round Robin scheduler.

    A task runs for at most *time_quantum* ticks before being preempted
    and moved to the back of the ready queue.  The simulation engine
    calls check_preemption() each tick; when it returns True the engine
    preempts the task and re-adds it via add_task().

    Args:
        time_quantum: Maximum consecutive ticks before preemption.
    """

    def __init__(self, time_quantum: int = 2) -> None:
        if time_quantum <= 0:
            raise ValueError(f"time_quantum must be positive, got {time_quantum}")
        self.time_quantum: int = time_quantum
        self._queue: deque[Task] = deque()

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int) -> Optional[Task]:
        if self._queue:
            return self._queue.popleft()
        return None

    def on_task_complete(self, task: Task) -> None:
        pass

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0

    def check_preemption(
        self,
        core_id: int,
        task: Task,
        ticks_on_core: int,
        current_time: int,
    ) -> bool:
        """Preempt when the task has exhausted its time quantum."""
        return ticks_on_core >= self.time_quantum
