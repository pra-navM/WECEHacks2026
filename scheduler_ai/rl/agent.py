"""RL agent that learns a scheduling policy and implements SchedulerBase."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from simulator.scheduler_base import SchedulerBase
from simulator.task import Task


def _get_action_mask(queue_len: int, max_queue_len: int) -> List[bool]:
    """Boolean mask: True for valid task indices."""
    mask = [False] * max_queue_len
    for i in range(min(queue_len, max_queue_len)):
        mask[i] = True
    return mask


class RLScheduler(SchedulerBase):
    """Scheduler that uses a trained SB3 policy (e.g. MaskablePPO) for get_next_task."""

    def __init__(
        self,
        model: Any,
        num_cores: int = 2,
        max_queue_len: int = 32,
    ) -> None:
        """Create an RL scheduler that uses the given SB3 model for decisions.

        Args:
            model: Loaded stable-baselines3 model (e.g. MaskablePPO.load(path)).
            num_cores: Number of cores (for observation).
            max_queue_len: Max queue size (must match training env).
        """
        self._model = model
        self._num_cores = num_cores
        self._max_queue_len = max_queue_len
        self._queue: List[Task] = []

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int) -> Optional[Task]:
        if not self._queue:
            return None

        from rl.env import build_obs_for_agent

        obs = build_obs_for_agent(
            current_time,
            self._num_cores,
            self._queue,
            self._max_queue_len,
        )
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        action_mask = _get_action_mask(len(self._queue), self._max_queue_len)
        try:
            action, _ = self._model.predict(obs, action_masks=[action_mask], deterministic=True)
        except TypeError:
            action, _ = self._model.predict(obs, deterministic=True)
        if hasattr(action, "item"):
            action = int(action.item())
        elif isinstance(action, (list, tuple, np.ndarray)):
            action = int(action[0]) if len(action) else 0
        else:
            action = int(action)

        if action < 0 or action >= len(self._queue):
            action = 0
        return self._queue.pop(action)

    def on_task_complete(self, task: Task) -> None:
        pass

    def on_task_blocked(self, task: Task) -> None:
        pass

    def on_task_unblocked(self, task: Task) -> None:
        self.add_task(task)

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0
