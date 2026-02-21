"""Task model for the CPU scheduling simulator."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class TickResult(Enum):
    """Outcome of a single tick of task execution."""

    RUNNING = auto()
    COMPLETED = auto()
    IO_BLOCKED = auto()


@dataclass(frozen=True)
class TaskMetadata:
    """Static resource profile attached to a task at creation time.

    All fields have safe defaults that reproduce Phase 1 behavior
    (no I/O blocking, no memory pressure, no affinity).
    """

    memory_rss: int = 1
    cache_locality: float = 0.0
    io_blocking_prob: float = 0.0
    buffer_requirement: int = 0


class Task:
    """A unit of work to be scheduled and executed on a CPU core.

    Each task arrives at a specific time, requires a fixed amount of CPU time
    (burst_time), and tracks its own execution progress.  An optional
    TaskMetadata packet describes resource requirements that influence
    scheduling quality.
    """

    __slots__ = (
        "task_id",
        "arrival_time",
        "burst_time",
        "remaining_time",
        "start_time",
        "completion_time",
        "metadata",
        "last_core_id",
        "io_block_remaining",
    )

    def __init__(
        self,
        task_id: int,
        arrival_time: int,
        burst_time: int,
        metadata: TaskMetadata | None = None,
    ) -> None:
        if burst_time <= 0:
            raise ValueError(f"burst_time must be positive, got {burst_time}")
        if arrival_time < 0:
            raise ValueError(f"arrival_time must be non-negative, got {arrival_time}")

        self.task_id: int = task_id
        self.arrival_time: int = arrival_time
        self.burst_time: int = burst_time
        self.remaining_time: int = burst_time
        self.start_time: Optional[int] = None
        self.completion_time: Optional[int] = None
        self.metadata: TaskMetadata = metadata or TaskMetadata()
        self.last_core_id: Optional[int] = None
        self.io_block_remaining: int = 0

    def is_complete(self) -> bool:
        """Return True if the task has finished execution."""
        return self.remaining_time == 0

    def run_one_tick(
        self,
        current_time: int,
        rng: random.Random | None = None,
    ) -> TickResult:
        """Execute this task for one time unit.

        Before doing real work the task rolls against its I/O blocking
        probability.  If the roll triggers, the task enters an I/O wait
        state (no progress is made) and the caller should remove it from
        the core.

        Args:
            current_time: The current simulation clock value.
            rng: Deterministic RNG for I/O blocking rolls.  When *None*
                 I/O blocking is disabled (Phase 1 compatible).

        Returns:
            TickResult indicating what happened this tick.

        Raises:
            RuntimeError: If called on an already-completed task.
        """
        if self.is_complete():
            raise RuntimeError(
                f"Task {self.task_id} is already complete; cannot run another tick."
            )

        if self.start_time is None:
            self.start_time = current_time

        if (
            rng is not None
            and self.metadata.io_blocking_prob > 0.0
            and rng.random() < self.metadata.io_blocking_prob
        ):
            self.io_block_remaining = rng.randint(1, 5)
            return TickResult.IO_BLOCKED

        self.remaining_time -= 1

        if self.remaining_time == 0:
            self.completion_time = current_time + 1
            return TickResult.COMPLETED

        return TickResult.RUNNING

    def tick_io_block(self) -> bool:
        """Advance the I/O block timer by one tick.

        Returns:
            True if the task has finished its I/O wait and is ready to
            be re-scheduled, False if still blocked.
        """
        if self.io_block_remaining > 0:
            self.io_block_remaining -= 1
        return self.io_block_remaining == 0

    @property
    def turnaround_time(self) -> Optional[int]:
        """Time from arrival to completion, or None if not yet complete."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    @property
    def waiting_time(self) -> Optional[int]:
        """Time spent waiting (turnaround minus burst), or None if not yet complete."""
        if self.turnaround_time is None:
            return None
        return self.turnaround_time - self.burst_time

    def __repr__(self) -> str:
        return (
            f"Task(id={self.task_id}, arrival={self.arrival_time}, "
            f"burst={self.burst_time}, remaining={self.remaining_time}, "
            f"start={self.start_time}, completion={self.completion_time})"
        )
