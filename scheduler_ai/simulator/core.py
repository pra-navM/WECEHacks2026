"""CPU core model for the scheduling simulator."""

from __future__ import annotations

import random
from typing import Optional

from simulator.task import Task, TickResult


class Core:
    """A single CPU core that executes one task at a time.

    Supports context-switch latency: when a new task is assigned, the core
    spends *context_switch_cost* ticks doing no useful work before the task
    begins executing.

    Args:
        core_id: Unique identifier for this core.
        context_switch_cost: Dead ticks incurred every time a task is assigned.
    """

    __slots__ = (
        "core_id",
        "current_task",
        "context_switch_cost",
        "context_switch_remaining",
        "ticks_on_core",
    )

    def __init__(self, core_id: int, context_switch_cost: int = 0) -> None:
        self.core_id: int = core_id
        self.current_task: Optional[Task] = None
        self.context_switch_cost: int = context_switch_cost
        self.context_switch_remaining: int = 0
        self.ticks_on_core: int = 0

    def assign_task(self, task: Task) -> None:
        """Assign a task to this core, triggering a context switch.

        Args:
            task: The task to execute.

        Raises:
            RuntimeError: If the core already has an assigned task.
        """
        if self.current_task is not None:
            raise RuntimeError(
                f"Core {self.core_id} is busy with Task {self.current_task.task_id}; "
                f"cannot assign Task {task.task_id}."
            )
        self.current_task = task
        self.context_switch_remaining = self.context_switch_cost
        self.ticks_on_core = 0
        task.last_core_id = self.core_id

    def tick(
        self,
        current_time: int,
        rng: random.Random | None = None,
    ) -> tuple[Optional[Task], Optional[TickResult]]:
        """Advance the core by one time unit.

        If a context switch is in progress, one dead tick is consumed.
        Otherwise the current task executes for one tick.

        Args:
            current_time: The current simulation clock value.
            rng: Deterministic RNG forwarded to the task for I/O rolls.

        Returns:
            A (completed_task, tick_result) pair.
            - When idle or switching: (None, None).
            - When the task completes: (task, TickResult.COMPLETED).
            - When the task blocks on I/O: (task, TickResult.IO_BLOCKED)
              and the task is detached from the core.
            - Otherwise: (None, TickResult.RUNNING).
        """
        if self.current_task is None:
            return None, None

        if self.context_switch_remaining > 0:
            self.context_switch_remaining -= 1
            return None, None

        result = self.current_task.run_one_tick(current_time, rng)
        self.ticks_on_core += 1

        if result is TickResult.COMPLETED:
            completed = self.current_task
            self.current_task = None
            self.ticks_on_core = 0
            return completed, TickResult.COMPLETED

        if result is TickResult.IO_BLOCKED:
            blocked = self.current_task
            self.current_task = None
            self.ticks_on_core = 0
            return blocked, TickResult.IO_BLOCKED

        return None, TickResult.RUNNING

    def preempt(self) -> Optional[Task]:
        """Remove and return the current task without completing it.

        The context-switch cost for the *next* task is incurred in
        the subsequent assign_task() call, not here.

        Returns:
            The preempted Task, or None if the core was idle.
        """
        task = self.current_task
        self.current_task = None
        self.ticks_on_core = 0
        self.context_switch_remaining = 0
        return task

    def is_idle(self) -> bool:
        """Return True if no task is currently assigned."""
        return self.current_task is None

    def __repr__(self) -> str:
        task_info = self.current_task.task_id if self.current_task else "idle"
        return f"Core(id={self.core_id}, task={task_info})"
