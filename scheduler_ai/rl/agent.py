"""RL agent that learns a scheduling policy and implements SchedulerBase."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from simulator.scheduler_base import SchedulerBase
from simulator.task import Task
import onnxruntime as ort



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

    def get_next_task(self, current_time: int, **kwargs: object) -> Optional[Task]:
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


def _governor_score_task_core(
    task: Task,
    core_id: int,
    core_heat: float,
    weights: np.ndarray,
    num_cores: int,
) -> float:
    """Core_Score = (1/len)*W_Length + Affinity*W_Affinity + (1-Heat)*W_Thermal."""
    w_length, w_affinity, w_thermal = weights[0], weights[1], weights[2]
    inv_len = 1.0 / max(1, task.remaining_time)
    affinity = 1.0 if (task.last_core_id is not None and task.last_core_id == core_id) else 0.0
    cool = 1.0 - core_heat
    return inv_len * w_length + affinity * w_affinity + cool * w_thermal


class GovernorScheduler(SchedulerBase):
    """Scheduler that uses a trained PPO governor (Box(3) weights) to score task-core pairs."""

    def __init__(
        self,
        model: Any,
        num_cores: int = 2,
        max_queue_len: int = 32,
    ) -> None:
        self._model = model
        self._num_cores = num_cores
        self._max_queue_len = max_queue_len
        self._queue: List[Task] = []

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int, **kwargs: object) -> Optional[Task]:
        if not self._queue:
            return None
        core_id = kwargs.get("core_id", 0)
        core_heat = kwargs.get("core_heat", 0.0)

        from rl.env import _build_governor_obs

        # Build minimal obs for governor (we only need the action = weights)
        idle_core_ids = [core_id]
        core_heats = [0.0] * self._num_cores
        if core_id < len(core_heats):
            core_heats[core_id] = core_heat
        obs = _build_governor_obs(
            current_time,
            self._num_cores,
            idle_core_ids,
            core_heats,
            self._queue,
            self._max_queue_len,
        )
        obs = np.array(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self._model.predict(obs, deterministic=True)
        weights = np.clip(np.asarray(action).ravel()[:3], -1.0, 1.0)

        best_score = -1e9
        best_i = 0
        for i, task in enumerate(self._queue):
            sc = _governor_score_task_core(task, core_id, core_heat, weights, self._num_cores)
            if sc > best_score:
                best_score = sc
                best_i = i
        return self._queue.pop(best_i)

    def on_task_complete(self, task: Task) -> None:
        pass

    def on_task_blocked(self, task: Task) -> None:
        pass

    def on_task_unblocked(self, task: Task) -> None:
        self.add_task(task)

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0

class QuantizedGovernor(SchedulerBase):
    """
    A production-grade version of GovernorScheduler using INT8 ONNX.
    Zero dependency on SB3 or PyTorch.
    """
    def __init__(self, model_path: str, num_cores: int = 2, max_queue_len: int = 32):
        self._num_cores = num_cores
        self._max_queue_len = max_queue_len
        self._queue: List[Task] = []
        
        # Initialize the high-speed ONNX session
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int, **kwargs: object) -> Optional[Task]:
        if not self._queue:
            return None
            
        core_id = kwargs.get("core_id", 0)
        core_heat = kwargs.get("core_heat", 0.0)

        # Build obs exactly like the original, but without importing heavy modules
        from rl.env import _build_governor_obs
        idle_core_ids = [core_id]
        core_heats = [0.0] * self._num_cores
        if core_id < len(core_heats):
            core_heats[core_id] = core_heat
            
        obs = _build_governor_obs(current_time, self._num_cores, idle_core_ids, 
                                 core_heats, self._queue, self._max_queue_len)
        
        # ONNX Inference
        obs_np = np.array(obs, dtype=np.float32).reshape(1, -1)
        action = self._session.run(None, {self._input_name: obs_np})[0]
        
        # action here is the 'weights' vector
        weights = np.clip(np.asarray(action).ravel()[:3], -1.0, 1.0)

        best_score = -1e9
        best_i = 0
        for i, task in enumerate(self._queue):
            sc = _governor_score_task_core(task, core_id, core_heat, weights, self._num_cores)
            if sc > best_score:
                best_score = sc
                best_i = i
        return self._queue.pop(best_i)

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0

    def on_task_unblocked(self, task: Task) -> None:
        self.add_task(task)
    
    def on_task_complete(self, task: Task) -> None: pass
    def on_task_blocked(self, task: Task) -> None: pass