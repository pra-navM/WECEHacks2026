"""Tick-based multicore simulation engine.

This module contains no scheduling policy logic. It orchestrates the
clock, task arrivals, core execution, preemption, I/O blocking, and
scheduler callbacks.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from simulator.core import Core
from simulator.scheduler_base import SchedulerBase
from simulator.task import Task, TickResult


class Simulation:
    """Deterministic, tick-driven multicore CPU simulation.

    Each tick the engine:
      1. Admits tasks whose arrival_time equals the current time.
      2. Advances I/O-blocked tasks and unblocks those that are ready.
      3. Checks each busy core for preemption (delegated to scheduler).
      4. Ticks every core (context-switch overhead or real execution).
      5. Handles completions and new I/O blocks.
      6. Dispatches ready tasks to idle cores via the scheduler.

    All scheduling decisions are delegated to the provided SchedulerBase
    implementation -- no policy logic lives here.

    Args:
        num_cores: Number of CPU cores to simulate.
        scheduler: The scheduling policy to use.
        tasks: The complete workload (need not be sorted).
        context_switch_cost: Dead ticks per context switch (default 0).
        seed: RNG seed for deterministic I/O blocking rolls.
    """

    def __init__(
        self,
        num_cores: int,
        scheduler: SchedulerBase,
        tasks: List[Task],
        context_switch_cost: int = 0,
        seed: int = 0,
    ) -> None:
        if num_cores <= 0:
            raise ValueError(f"num_cores must be positive, got {num_cores}")

        self._cores: List[Core] = [
            Core(i, context_switch_cost=context_switch_cost) for i in range(num_cores)
        ]
        self._scheduler: SchedulerBase = scheduler
        self._tasks: List[Task] = sorted(tasks, key=lambda t: t.arrival_time)
        self._current_time: int = 0
        self._rng: random.Random = random.Random(seed)
        self._blocked_tasks: List[Task] = []
        self._task_cursor: int = 0
        self._completed_count: int = 0

    def step_tick(
        self,
    ) -> Tuple[List[int], List[Task], bool, dict]:
        """Run one tick (phases 1-4 only). Caller must dispatch to idle cores.

        Returns:
            idle_core_ids: Core indices that are idle after this tick.
            completed_this_tick: Tasks that completed this tick.
            done: True if all tasks have completed.
            info: Dict with e.g. 'current_time' for debugging.
        """
        total_tasks = len(self._tasks)
        completed_this_tick: List[Task] = []

        # 1. Admit newly arrived tasks.
        while (
            self._task_cursor < total_tasks
            and self._tasks[self._task_cursor].arrival_time <= self._current_time
        ):
            self._scheduler.add_task(self._tasks[self._task_cursor])
            self._task_cursor += 1

        # 2. Advance I/O-blocked tasks.
        still_blocked: List[Task] = []
        for task in self._blocked_tasks:
            if task.tick_io_block():
                self._scheduler.on_task_unblocked(task)
            else:
                still_blocked.append(task)
        self._blocked_tasks = still_blocked

        # 3. Check preemption on busy cores.
        for core in self._cores:
            if core.current_task is not None and core.context_switch_remaining == 0:
                if self._scheduler.check_preemption(
                    core.core_id,
                    core.current_task,
                    core.ticks_on_core,
                    self._current_time,
                ):
                    preempted = core.preempt()
                    if preempted is not None:
                        self._scheduler.add_task(preempted)

        # 4. Tick every core.
        for core in self._cores:
            task, result = core.tick(self._current_time, self._rng)

            if result is TickResult.COMPLETED:
                assert task is not None
                self._scheduler.on_task_complete(task)
                self._completed_count += 1
                completed_this_tick.append(task)
            elif result is TickResult.IO_BLOCKED:
                assert task is not None
                self._scheduler.on_task_blocked(task)
                self._blocked_tasks.append(task)

        self._current_time += 1

        idle_core_ids = [c.core_id for c in self._cores if c.is_idle()]
        done = self._completed_count >= total_tasks
        info: dict = {"current_time": self._current_time - 1}

        return idle_core_ids, completed_this_tick, done, info

    def dispatch(self, core_id: int, task: Task) -> None:
        """Assign a task to a core. Used when env drives dispatch (no scheduler call)."""
        self._cores[core_id].assign_task(task)

    def run(self) -> List[Task]:
        """Execute the full simulation and return the completed task list.

        Returns:
            The original task objects, now populated with start_time,
            completion_time, and remaining_time == 0.
        """
        task_cursor = 0
        total_tasks = len(self._tasks)
        completed_count = 0

        while completed_count < total_tasks:
            # 1. Admit newly arrived tasks.
            while (
                task_cursor < total_tasks
                and self._tasks[task_cursor].arrival_time <= self._current_time
            ):
                self._scheduler.add_task(self._tasks[task_cursor])
                task_cursor += 1

            # 2. Advance I/O-blocked tasks.
            still_blocked: List[Task] = []
            for task in self._blocked_tasks:
                if task.tick_io_block():
                    self._scheduler.on_task_unblocked(task)
                else:
                    still_blocked.append(task)
            self._blocked_tasks = still_blocked

            # 3. Check preemption on busy cores.
            for core in self._cores:
                if core.current_task is not None and core.context_switch_remaining == 0:
                    if self._scheduler.check_preemption(
                        core.core_id,
                        core.current_task,
                        core.ticks_on_core,
                        self._current_time,
                    ):
                        preempted = core.preempt()
                        if preempted is not None:
                            self._scheduler.add_task(preempted)

            # 4. Tick every core.
            for core in self._cores:
                task, result = core.tick(self._current_time, self._rng)

                if result is TickResult.COMPLETED:
                    assert task is not None
                    self._scheduler.on_task_complete(task)
                    completed_count += 1
                elif result is TickResult.IO_BLOCKED:
                    assert task is not None
                    self._scheduler.on_task_blocked(task)
                    self._blocked_tasks.append(task)

            # 5. Dispatch tasks to idle cores.
            for core in self._cores:
                if core.is_idle():
                    next_task = self._scheduler.get_next_task(self._current_time)
                    if next_task is not None:
                        core.assign_task(next_task)

            self._current_time += 1

        return self._tasks
