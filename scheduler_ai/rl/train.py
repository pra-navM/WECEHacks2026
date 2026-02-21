"""Training loop for the RL scheduling agent.

This module will orchestrate:
  - Episode generation via SchedulerEnv
  - Policy updates via the chosen RL algorithm
  - Logging and checkpointing
"""

from __future__ import annotations


def train() -> None:
    """Run the RL training loop."""
    raise NotImplementedError("Phase 3: RL training")
