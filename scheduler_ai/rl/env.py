"""Gymnasium-compatible environment wrapping the simulation engine for RL scheduling."""

from __future__ import annotations

import random
from typing import Any, List, Optional, Tuple

import numpy as np

from simulator.scheduler_base import SchedulerBase
from simulator.simulation import Simulation
from simulator.task import Task
from workload.generator import generate_workload, generate_workload_mixed

# Normalization constants for observation (fixed range for generalization)
MAX_TIME = 500.0
MAX_BURST = 100.0


class _QueueOnlyScheduler(SchedulerBase):
    """Minimal scheduler that only holds the ready queue. Used by SchedulerEnv."""

    def __init__(self) -> None:
        self._queue: List[Task] = []

    def add_task(self, task: Task) -> None:
        self._queue.append(task)

    def get_next_task(self, current_time: int) -> Optional[Task]:
        return None  # Env drives dispatch via step(); never called for dispatch

    def on_task_complete(self, task: Task) -> None:
        pass

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0

    def get_queue(self) -> List[Task]:
        return self._queue

    def remove_and_get_at(self, index: int) -> Task:
        return self._queue.pop(index)


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


def _build_obs(
    current_time: int,
    num_cores: int,
    num_idle: int,
    queue: List[Task],
    max_queue_len: int,
) -> np.ndarray:
    """Build fixed-size observation vector."""
    # Global: current_time (norm), num_idle_cores, num_ready_tasks, num_cores
    t_norm = min(1.0, current_time / MAX_TIME)
    n_ready = min(len(queue), max_queue_len)
    global_part = np.array(
        [t_norm, num_idle / max(1, num_cores), n_ready / max_queue_len, num_cores / 8.0],
        dtype=np.float32,
    )

    # Per-task slots: remaining_time, arrival_time, burst_time, wait_time_so_far, io_blocking_prob
    task_dim = 5
    slot_values = []
    for i in range(max_queue_len):
        if i < len(queue):
            t = queue[i]
            executed = t.burst_time - t.remaining_time
            wait_so_far = max(0, current_time - t.arrival_time - executed)
            slot_values.extend(
                [
                    t.remaining_time / MAX_BURST,
                    t.arrival_time / MAX_TIME,
                    t.burst_time / MAX_BURST,
                    wait_so_far / MAX_TIME,
                    t.metadata.io_blocking_prob,
                ]
            )
        else:
            slot_values.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    task_part = np.array(slot_values, dtype=np.float32)
    obs = np.concatenate([global_part, task_part])
    return obs


def _get_action_mask(queue_len: int, max_queue_len: int) -> np.ndarray:
    """Boolean mask: True for valid task indices."""
    mask = np.zeros(max_queue_len, dtype=bool)
    for i in range(min(queue_len, max_queue_len)):
        mask[i] = True
    return mask


class SchedulerEnv(gym.Env if gym else object):
    """Gymnasium-style environment for RL-based CPU scheduling."""

    def __init__(
        self,
        num_cores: int = 2,
        max_queue_len: int = 32,
        context_switch_cost: int = 0,
        seed: Optional[int] = None,
        workload_type: str = "mixed",
        num_tasks_range: Tuple[int, int] = (20, 50),
        episode_timeout: int = 2000,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for SchedulerEnv. Install with: pip install gymnasium")

        self.num_cores = num_cores
        self.max_queue_len = max_queue_len
        self.context_switch_cost = context_switch_cost
        self.workload_type = workload_type
        self.num_tasks_range = num_tasks_range
        self.episode_timeout = episode_timeout

        obs_dim = 4 + max_queue_len * 5
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(max_queue_len)

        self._rng = random.Random(seed)
        self._sim: Optional[Simulation] = None
        self._scheduler: Optional[_QueueOnlyScheduler] = None
        self._idle_core_ids: List[int] = []
        self._steps_this_episode = 0

    def _make_workload(self, seed: Optional[int] = None) -> List[Task]:
        n = self._rng.randint(*self.num_tasks_range)
        s = seed if seed is not None else self._rng.randint(0, 2**31 - 1)
        if self.workload_type == "mixed":
            return generate_workload_mixed(num_tasks=n, seed=s)
        return generate_workload(num_tasks=n, seed=s)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = random.Random(seed)

        tasks = self._make_workload(seed=seed)
        self._scheduler = _QueueOnlyScheduler()
        self._sim = Simulation(
            num_cores=self.num_cores,
            scheduler=self._scheduler,
            tasks=tasks,
            context_switch_cost=self.context_switch_cost,
            seed=self._rng.randint(0, 2**31 - 1),
        )
        self._idle_core_ids = []
        self._steps_this_episode = 0

        # Run step_tick until we need a dispatch (idle core + non-empty queue) or done
        while True:
            idle_core_ids, completed, done, info = self._sim.step_tick()
            self._idle_core_ids = idle_core_ids
            if done:
                break
            if idle_core_ids and self._scheduler.has_pending_tasks():
                break
            if self._sim._current_time >= self.episode_timeout:
                done = True
                break

        queue = self._scheduler.get_queue()
        obs = _build_obs(
            self._sim._current_time - 1,
            self.num_cores,
            len(self._idle_core_ids),
            queue,
            self.max_queue_len,
        )
        action_mask = _get_action_mask(len(queue), self.max_queue_len)
        info = {"action_mask": action_mask}
        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._sim is None or self._scheduler is None:
            raise RuntimeError("Call reset() first")

        reward = 0.0
        queue = self._scheduler.get_queue()

        # Map action to task and dispatch to first idle core
        if self._idle_core_ids and queue and 0 <= action < len(queue):
            task = self._scheduler.remove_and_get_at(action)
            core_id = self._idle_core_ids.pop(0)
            self._sim.dispatch(core_id, task)

        # If more idle cores and queue non-empty, return same tick (need another action)
        if self._idle_core_ids and self._scheduler.has_pending_tasks():
            queue = self._scheduler.get_queue()
            obs = _build_obs(
                self._sim._current_time - 1,
                self.num_cores,
                len(self._idle_core_ids),
                queue,
                self.max_queue_len,
            )
            action_mask = _get_action_mask(len(queue), self.max_queue_len)
            return obs, 0.0, False, False, {"action_mask": action_mask}

        # No more dispatches this tick; advance one tick
        idle_core_ids, completed_this_tick, done, info = self._sim.step_tick()
        self._idle_core_ids = idle_core_ids
        self._steps_this_episode += 1

        for t in completed_this_tick:
            if t.turnaround_time is not None:
                reward -= float(t.turnaround_time)

        truncated = self._sim._current_time >= self.episode_timeout and not done

        # Build next obs (may need more dispatches or next tick)
        while not done and not truncated:
            if self._idle_core_ids and self._scheduler.has_pending_tasks():
                queue = self._scheduler.get_queue()
                obs = _build_obs(
                    self._sim._current_time - 1,
                    self.num_cores,
                    len(self._idle_core_ids),
                    queue,
                    self.max_queue_len,
                )
                action_mask = _get_action_mask(len(queue), self.max_queue_len)
                return obs, reward, False, False, {"action_mask": action_mask}
            idle_core_ids, completed_this_tick, done, info = self._sim.step_tick()
            self._idle_core_ids = idle_core_ids
            self._steps_this_episode += 1
            for t in completed_this_tick:
                if t.turnaround_time is not None:
                    reward -= float(t.turnaround_time)
            truncated = self._sim._current_time >= self.episode_timeout and not done

        # Episode ended or truncated
        queue = self._scheduler.get_queue()
        obs = _build_obs(
            self._sim._current_time - 1,
            self.num_cores,
            len(self._idle_core_ids),
            queue,
            self.max_queue_len,
        )
        action_mask = _get_action_mask(len(queue), self.max_queue_len)
        return obs, reward, done, truncated, {"action_mask": action_mask}

    def action_masks(self) -> np.ndarray:
        """Return action mask for sb3_contrib MaskablePPO."""
        if self._scheduler is None:
            return np.zeros(self.max_queue_len, dtype=bool)
        return _get_action_mask(len(self._scheduler.get_queue()), self.max_queue_len)


def build_obs_for_agent(
    current_time: int,
    num_cores: int,
    queue: List[Task],
    max_queue_len: int,
) -> np.ndarray:
    """Build observation vector (same encoding as env). Used by RLScheduler at inference."""
    num_idle = 0  # Agent doesn't have core state; use 0 or pass if available
    return _build_obs(current_time, num_cores, num_idle, queue, max_queue_len)


# --- Governor (hardware-aware) env: Box(3) action = [W_Length, W_Affinity, W_Thermal] ---
CONTEXT_SWITCH_COST_DEFAULT = 7
CACHE_LOCALITY_COST = 15


def _build_governor_obs(
    current_time: int,
    num_cores: int,
    idle_core_ids: List[int],
    core_heats: List[float],
    queue: List[Task],
    max_queue_len: int,
) -> np.ndarray:
    """Observation for governor: global + per-core heat + per-task (remaining, last_core)."""
    t_norm = min(1.0, current_time / MAX_TIME)
    n_ready = min(len(queue), max_queue_len)
    global_part = np.array(
        [t_norm, len(idle_core_ids) / max(1, num_cores), n_ready / max_queue_len, num_cores / 8.0],
        dtype=np.float32,
    )
    # Per-core heat (for all cores so size is fixed)
    heat_part = np.array([core_heats[i] if i < len(core_heats) else 0.0 for i in range(num_cores)], dtype=np.float32)
    # Per-task: remaining_time, last_core_id (normalized 0-1, or -1 if none)
    task_part = []
    for i in range(max_queue_len):
        if i < len(queue):
            t = queue[i]
            last = (t.last_core_id / num_cores) if t.last_core_id is not None else -1.0
            task_part.extend([t.remaining_time / MAX_BURST, t.arrival_time / MAX_TIME, t.burst_time / MAX_BURST, last])
        else:
            task_part.extend([0.0, 0.0, 0.0, -1.0])
    task_part = np.array(task_part, dtype=np.float32)
    return np.concatenate([global_part, heat_part, task_part])


def _governor_score_task_core(
    task: Task,
    core_id: int,
    core_heat: float,
    weights: np.ndarray,
    num_cores: int,
) -> float:
    """Core_Score = (1/len)*W_Length + Affinity*W_Affinity + (1-Heat)*W_Thermal."""
    w_length, w_affinity, w_thermal = weights[0], weights[1], weights[2]
    inv_len = 1.0 / max(1, task.remaining_time)  # prefer short
    affinity = 1.0 if (task.last_core_id is not None and task.last_core_id == core_id) else 0.0
    cool = 1.0 - core_heat  # prefer cool cores
    return inv_len * w_length + affinity * w_affinity + cool * w_thermal


class GovernorSchedulerEnv(gym.Env if gym else object):
    """Governor-style env: action Box(3) = [W_Length, W_Affinity, W_Thermal], hardware friction."""

    def __init__(
        self,
        num_cores: int = 2,
        max_queue_len: int = 32,
        context_switch_cost: int = CONTEXT_SWITCH_COST_DEFAULT,
        cache_locality_cost: int = CACHE_LOCALITY_COST,
        seed: Optional[int] = None,
        workload_type: str = "mixed",
        num_tasks_range: Tuple[int, int] = (20, 50),
        episode_timeout: int = 2000,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium")
        self.num_cores = num_cores
        self.max_queue_len = max_queue_len
        self.context_switch_cost = context_switch_cost
        self.cache_locality_cost = cache_locality_cost
        self.workload_type = workload_type
        self.num_tasks_range = num_tasks_range
        self.episode_timeout = episode_timeout

        obs_dim = 4 + num_cores + max_queue_len * 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )  # [W_Length, W_Affinity, W_Thermal]

        self._rng = random.Random(seed)
        self._sim: Optional[Simulation] = None
        self._scheduler: Optional[_QueueOnlyScheduler] = None
        self._idle_core_ids: List[int] = []
        self._episode_cs_penalty: float = 0.0
        self._episode_throttling: int = 0
        self._episode_waits: List[float] = []

    def _make_workload(self, seed: Optional[int] = None) -> List[Task]:
        n = self._rng.randint(*self.num_tasks_range)
        s = seed if seed is not None else self._rng.randint(0, 2**31 - 1)
        if self.workload_type == "mixed":
            return generate_workload_mixed(num_tasks=n, seed=s)
        return generate_workload(num_tasks=n, seed=s)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = random.Random(seed)
        tasks = self._make_workload(seed=seed)
        self._scheduler = _QueueOnlyScheduler()
        self._sim = Simulation(
            num_cores=self.num_cores,
            scheduler=self._scheduler,
            tasks=tasks,
            context_switch_cost=self.context_switch_cost,
            cache_locality_cost=self.cache_locality_cost,
            seed=self._rng.randint(0, 2**31 - 1),
        )
        self._idle_core_ids = []
        self._episode_cs_penalty = 0.0
        self._episode_throttling = 0
        self._episode_waits = []

        while True:
            idle_core_ids, completed, done, info = self._sim.step_tick()
            self._idle_core_ids = idle_core_ids
            self._episode_throttling += info.get("throttling_events", 0)
            for t in completed:
                if t.waiting_time is not None:
                    self._episode_waits.append(float(t.waiting_time))
            if done:
                break
            if idle_core_ids and self._scheduler.has_pending_tasks():
                break
            if self._sim._current_time >= self.episode_timeout:
                done = True
                break

        core_heats = [self._sim._cores[i].get_heat_normalized() for i in range(self.num_cores)]
        queue = self._scheduler.get_queue()
        obs = _build_governor_obs(
            self._sim._current_time - 1,
            self.num_cores,
            self._idle_core_ids,
            core_heats,
            queue,
            self.max_queue_len,
        )
        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._sim is None or self._scheduler is None:
            raise RuntimeError("Call reset() first")

        reward = 0.0
        weights = np.clip(np.asarray(action, dtype=np.float32).ravel()[:3], -1.0, 1.0)
        queue = self._scheduler.get_queue()
        core_heats = [self._sim._cores[i].get_heat_normalized() for i in range(self.num_cores)]

        # Assign (task, core) greedily by governor score
        while self._idle_core_ids and self._scheduler.has_pending_tasks():
            q = self._scheduler.get_queue()
            if not q:
                break
            best_score = -1e9
            best_qi, best_cid = 0, self._idle_core_ids[0]
            for qi, task in enumerate(q):
                for cid in self._idle_core_ids:
                    h = core_heats[cid]
                    sc = _governor_score_task_core(task, cid, h, weights, self.num_cores)
                    if sc > best_score:
                        best_score = sc
                        best_qi = qi
                        best_cid = cid
            task = self._scheduler.remove_and_get_at(best_qi)
            penalty = self._sim.dispatch(best_cid, task)
            self._episode_cs_penalty += penalty
            reward -= float(penalty)
            self._idle_core_ids = [c for c in self._idle_core_ids if c != best_cid]
            core_heats[best_cid] = self._sim._cores[best_cid].get_heat_normalized()

        # Advance one tick
        idle_core_ids, completed_this_tick, done, info = self._sim.step_tick()
        self._idle_core_ids = idle_core_ids
        te = info.get("throttling_events", 0)
        self._episode_throttling += te
        reward -= te * 100.0

        for t in completed_this_tick:
            if t.waiting_time is not None:
                self._episode_waits.append(float(t.waiting_time))

        truncated = self._sim._current_time >= self.episode_timeout and not done

        while not done and not truncated:
            if self._idle_core_ids and self._scheduler.has_pending_tasks():
                core_heats = [self._sim._cores[i].get_heat_normalized() for i in range(self.num_cores)]
                queue = self._scheduler.get_queue()
                obs = _build_governor_obs(
                    self._sim._current_time - 1,
                    self.num_cores,
                    self._idle_core_ids,
                    core_heats,
                    queue,
                    self.max_queue_len,
                )
                return obs, reward, False, False, {}
            idle_core_ids, completed_this_tick, done, info = self._sim.step_tick()
            self._idle_core_ids = idle_core_ids
            te = info.get("throttling_events", 0)
            self._episode_throttling += te
            reward -= te * 100.0
            for t in completed_this_tick:
                if t.waiting_time is not None:
                    self._episode_waits.append(float(t.waiting_time))
            truncated = self._sim._current_time >= self.episode_timeout and not done

        # Terminal: add -avg_wait (CS and throttling already added per-step)
        avg_wait = (sum(self._episode_waits) / len(self._episode_waits)) if self._episode_waits else 0.0
        reward -= avg_wait

        core_heats = [self._sim._cores[i].get_heat_normalized() for i in range(self.num_cores)]
        queue = self._scheduler.get_queue()
        obs = _build_governor_obs(
            self._sim._current_time - 1,
            self.num_cores,
            self._idle_core_ids,
            core_heats,
            queue,
            self.max_queue_len,
        )
        return obs, reward, done, truncated, {}
