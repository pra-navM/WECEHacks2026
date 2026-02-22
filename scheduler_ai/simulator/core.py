"""CPU core model for the scheduling simulator."""

from __future__ import annotations

import random
from typing import Optional, Tuple

from simulator.task import Task, TickResult

# Thermal: heat increases when busy, decreases when idle. If heat > 90%, 50% tick rate.
HEAT_INCREASE_PER_BUSY_TICK = 2.0
HEAT_DECREASE_PER_IDLE_TICK = 1.0
HEAT_THROTTLE_THRESHOLD = 90.0
HEAT_MAX = 100.0


class Core:
    """A single CPU core that executes one task at a time.

    Supports context-switch latency, cache locality cost (migration penalty),
    and thermal throttling (performance drop when heat > 90%).

    Args:
        core_id: Unique identifier for this core.
        context_switch_cost: Dead ticks incurred every time a task is assigned.
        cache_locality_cost: Extra dead ticks when task moves to a different core.
    """

    __slots__ = (
        "core_id",
        "current_task",
        "context_switch_cost",
        "cache_locality_cost",
        "context_switch_remaining",
        "ticks_on_core",
        "heat",
    )

    def __init__(
        self,
        core_id: int,
        context_switch_cost: int = 0,
        cache_locality_cost: int = 0,
    ) -> None:
        self.core_id: int = core_id
        self.current_task: Optional[Task] = None
        self.context_switch_cost: int = context_switch_cost
        self.cache_locality_cost: int = cache_locality_cost
        self.context_switch_remaining: int = 0
        self.ticks_on_core: int = 0
        self.heat: float = 0.0

    def assign_task(self, task: Task) -> int:
        """Assign a task to this core, triggering a context switch.

        Returns:
            Total penalty ticks (context_switch_cost + cache_locality_cost if migrated).

        Raises:
            RuntimeError: If the core already has an assigned task.
        """
        if self.current_task is not None:
            raise RuntimeError(
                f"Core {self.core_id} is busy with Task {self.current_task.task_id}; "
                f"cannot assign Task {task.task_id}."
            )
        migrated = task.last_core_id is not None and task.last_core_id != self.core_id
        extra = self.cache_locality_cost if migrated else 0
        self.context_switch_remaining = self.context_switch_cost + extra
        self.current_task = task
        self.ticks_on_core = 0
        task.last_core_id = self.core_id
        return self.context_switch_cost + extra

    def tick(
        self,
        current_time: int,
        rng: random.Random | None = None,
    ) -> Tuple[Optional[Task], Optional[TickResult], bool]:
        """Advance the core by one time unit.

        If a context switch is in progress, one dead tick is consumed.
        Otherwise the current task executes for one tick (or 0.5 effective when throttled).

        Returns:
            (completed_task, tick_result, was_throttled).
        """
        if self.current_task is None:
            self.heat = max(0.0, self.heat - HEAT_DECREASE_PER_IDLE_TICK)
            return None, None, False

        if self.context_switch_remaining > 0:
            self.context_switch_remaining -= 1
            return None, None, False

        # Thermal throttling: if heat > 90%, 50% effective tick rate
        was_throttled = self.heat > HEAT_THROTTLE_THRESHOLD
        if was_throttled and rng is not None and rng.random() < 0.5:
            self.heat = min(HEAT_MAX, self.heat + HEAT_INCREASE_PER_BUSY_TICK)
            return None, TickResult.RUNNING, True

        result = self.current_task.run_one_tick(current_time, rng)
        self.ticks_on_core += 1
        self.heat = min(HEAT_MAX, self.heat + HEAT_INCREASE_PER_BUSY_TICK)

        if result is TickResult.COMPLETED:
            completed = self.current_task
            self.current_task = None
            self.ticks_on_core = 0
            return completed, TickResult.COMPLETED, was_throttled

        if result is TickResult.IO_BLOCKED:
            blocked = self.current_task
            self.current_task = None
            self.ticks_on_core = 0
            return blocked, TickResult.IO_BLOCKED, was_throttled

        return None, TickResult.RUNNING, was_throttled

    def get_heat_normalized(self) -> float:
        """Return heat in [0, 1] for observations."""
        return self.heat / HEAT_MAX

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
