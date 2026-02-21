"""RL agent that learns a scheduling policy.

The agent will subclass SchedulerBase and implement get_next_task()
using a learned policy (e.g., DQN, PPO) instead of a heuristic.
"""

from __future__ import annotations


class RLAgent:
    """Reinforcement learning agent for adaptive scheduling."""

    def __init__(self) -> None:
        raise NotImplementedError("Phase 3: RL agent")
