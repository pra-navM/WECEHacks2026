"""Gymnasium-compatible environment wrapping the simulation engine.

This module will expose the simulation as an RL environment where:
  - Observation: queue state, core utilization, task attributes.
  - Action: which task to dispatch to an idle core.
  - Reward: derived from scheduling performance metrics.

The Simulation.run() tick loop will be refactored into a step()
interface to support episodic RL training.
"""

from __future__ import annotations


class SchedulerEnv:
    """Gymnasium-style environment for RL-based scheduling."""

    def __init__(self) -> None:
        raise NotImplementedError("Phase 3: RL environment")
