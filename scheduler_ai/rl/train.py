"""Training loop for the RL scheduling agent: Governor (Box(3)) + parallel envs."""

from __future__ import annotations

import argparse
import os

from rl.env import GovernorSchedulerEnv


def train(
    total_timesteps: int = 200_000,
    save_path: str = "rl/checkpoints/scheduler_pro_v1.zip",
    num_cores: int = 2,
    max_queue_len: int = 32,
    n_envs: int = 8,
    num_tasks_range: tuple[int, int] = (20, 50),
    context_switch_cost: int = 7,
    cache_locality_cost: int = 15,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int | None = None,
    device: str = "cpu",
) -> None:
    """Run the RL training loop with PPO + SubprocVecEnv (Governor, hardware friction)."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required for training. "
            "Install with: pip install stable-baselines3"
        ) from e

    def make_env():
        def _init():
            return GovernorSchedulerEnv(
                num_cores=num_cores,
                max_queue_len=max_queue_len,
                context_switch_cost=context_switch_cost,
                cache_locality_cost=cache_locality_cost,
                seed=seed,
                workload_type="mixed",
                num_tasks_range=num_tasks_range,
                episode_timeout=2000,
            )
        return _init

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        seed=seed,
        device=device,
    )

    # Progress bar callback
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_timesteps, desc="Training", unit="step")
    except ImportError:
        progress_bar = None

    class ProgressCallback(BaseCallback):
        def __init__(self, total: int, pbar=None):
            super().__init__()
            self.total = total
            self.pbar = pbar

        def _on_step(self) -> bool:
            if self.pbar is not None:
            # Update by the number of environments processed in this step
                self.pbar.update(self.training_env.num_envs) 
            return True

    callback = ProgressCallback(total_timesteps, progress_bar) if progress_bar else None

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    if progress_bar is not None:
        progress_bar.close()
    model.save(save_path)
    print(f"Model saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Governor RL scheduler (PPO, SubprocVecEnv, hardware friction)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="rl/checkpoints/scheduler_pro_v1.zip",
        help="Path to save the trained model (default: scheduler_pro_v1.zip)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=2,
        help="Number of CPU cores in simulation (default: 2)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs for SubprocVecEnv (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PPO (default: cpu)",
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_path=args.save,
        num_cores=args.cores,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
