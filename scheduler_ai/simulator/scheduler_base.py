"""Abstract base class for all scheduling policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from scheduler_ai.simulator.task import Task


class SchedulerBase(ABC):
    """Interface that every scheduling policy must implement.

    The simulation engine interacts with schedulers exclusively through
    these methods, keeping policy logic fully decoupled from the
    simulation loop.

    Subclasses *must* implement the four abstract methods.  The three
    concrete hook methods (check_preemption, on_task_blocked,
    on_task_unblocked) have safe defaults so that non-preemptive
    schedulers need not override them.
    """

    @abstractmethod
    def add_task(self, task: Task) -> None:
        """Accept a newly arrived task into the scheduler's ready queue.

        Args:
            task: The task that has just arrived.
        """

    @abstractmethod
    def get_next_task(self, current_time: int) -> Optional[Task]:
        """Select and remove the next task to run on an idle core.

        Args:
            current_time: The current simulation clock value.

        Returns:
            The chosen Task, or None if no task is ready.
        """

    @abstractmethod
    def on_task_complete(self, task: Task) -> None:
        """Notify the scheduler that a task has finished execution.

        This hook allows schedulers that need post-completion bookkeeping
        (e.g., Round Robin re-enqueue logic) to act accordingly.

        Args:
            task: The task that just completed.
        """

    @abstractmethod
    def has_pending_tasks(self) -> bool:
        """Return True if the scheduler still has tasks in its ready queue."""

    # ------------------------------------------------------------------
    # Concrete hooks with safe defaults (Phase 2)
    # ------------------------------------------------------------------

    def check_preemption(
        self,
        core_id: int,
        task: Task,
        ticks_on_core: int,
        current_time: int,
    ) -> bool:
        """Decide whether *task* on *core_id* should be preempted.

        Called once per busy core per tick, before the core executes.

        Args:
            core_id: Identifier of the core running the task.
            task: The currently executing task.
            ticks_on_core: How many ticks the task has run since last dispatch.
            current_time: The current simulation clock value.

        Returns:
            True to preempt the task (it will be re-added via add_task).
        """
        return False

    def on_task_blocked(self, task: Task) -> None:
        """Called when a running task enters an I/O wait.

        Default behaviour is a no-op.  Override if the scheduler needs
        to track blocked tasks separately.

        Args:
            task: The task that just blocked.
        """

    def on_task_unblocked(self, task: Task) -> None:
        """Called when a task finishes its I/O wait and is ready again.

        Default behaviour re-adds the task to the ready queue via
        add_task().

        Args:
            task: The task that is now ready.
        """
        self.add_task(task)
