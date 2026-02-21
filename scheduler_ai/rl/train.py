"""Training loop for the RL scheduling agent using MaskablePPO."""

from __future__ import annotations

import argparse
import os

from rl.env import SchedulerEnv


def train(
    total_timesteps: int = 200_000,
    save_path: str = "rl/checkpoints/ppo_scheduler.zip",
    num_cores: int = 2,
    max_queue_len: int = 32,
    num_tasks_range: tuple[int, int] = (20, 50),
    learning_rate: float = 1e-3,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int | None = None,
) -> None:
    """Run the RL training loop with MaskablePPO."""
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
    except ImportError as e:
        raise ImportError(
            "sb3_contrib and stable-baselines3 are required for training. "
            "Install with: pip install stable-baselines3 sb3-contrib"
        ) from e

    env = SchedulerEnv(
        num_cores=num_cores,
        max_queue_len=max_queue_len,
        context_switch_cost=0,
        seed=seed,
        workload_type="mixed",
        num_tasks_range=num_tasks_range,
        episode_timeout=2000,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        seed=seed,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL scheduling agent (MaskablePPO)")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="rl/checkpoints/ppo_scheduler.zip",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=2,
        help="Number of CPU cores (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_path=args.save,
        num_cores=args.cores,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
