"""Comparative analysis: RL scheduler vs Round Robin on bursty workload."""

from __future__ import annotations

import argparse
import copy

from metrics.performance import compute_metrics
from simulator.round_robin import RoundRobinScheduler
from simulator.simulation import Simulation
from simulator.task import Task
from workload.generator import generate_workload_mixed


def load_rl_scheduler(model_path: str, num_cores: int):
    """Load trained PPO model and return an RLScheduler."""
    import os
    from sb3_contrib import MaskablePPO
    from rl.agent import RLScheduler
    
    # Normalize path and ensure it doesn't have double .zip extension
    path = model_path
    if path.endswith(".zip.zip"):
        path = path[:-4]  # Remove one .zip
    elif not path.endswith(".zip"):
        path = path + ".zip"
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL model not found at '{path}'. "
            f"Please train a model first using:\n"
            f"  python -m rl.train --timesteps 200000 --save {path}"
        )
    
    model = MaskablePPO.load(path)
    return RLScheduler(model=model, num_cores=num_cores, max_queue_len=32)


def run_comparison(
    model_path: str = "rl/checkpoints/ppo_scheduler.zip",
    num_cores: int = 2,
    num_tasks: int = 40,
    workload_seed: int = 42,
    rr_quantum: int = 2,
    context_switch_cost: int = 0,
) -> None:
    """Run same workload with RR and RL, print side-by-side metrics."""
    workload = generate_workload_mixed(num_tasks=num_tasks, seed=workload_seed)

    # Round Robin
    workload_rr = copy.deepcopy(workload)
    rr_scheduler = RoundRobinScheduler(time_quantum=rr_quantum)
    sim_rr = Simulation(
        num_cores=num_cores,
        scheduler=rr_scheduler,
        tasks=workload_rr,
        context_switch_cost=context_switch_cost,
        seed=workload_seed,
    )
    completed_rr = sim_rr.run()
    metrics_rr = compute_metrics(completed_rr)

    # RL
    workload_rl = copy.deepcopy(workload)
    rl_scheduler = load_rl_scheduler(model_path, num_cores)
    sim_rl = Simulation(
        num_cores=num_cores,
        scheduler=rl_scheduler,
        tasks=workload_rl,
        context_switch_cost=context_switch_cost,
        seed=workload_seed,
    )
    completed_rl = sim_rl.run()
    metrics_rl = compute_metrics(completed_rl)

    # Report
    print("\n=== Comparative Analysis: RL vs Round Robin (bursty workload) ===\n")
    print(f"  Workload: {num_tasks} tasks, seed={workload_seed}, cores={num_cores}")
    print()
    print(f"  {'Metric':<22}  {'Round Robin':>12}  {'RL':>12}  {'Improvement':>12}")
    print("  " + "-" * 60)

    for key in ["avg_turnaround", "avg_waiting_time", "max_turnaround"]:
        rr_val = metrics_rr[key]
        rl_val = metrics_rl[key]
        if rr_val > 0:
            pct = 100.0 * (rr_val - rl_val) / rr_val
            imp = f"{pct:+.1f}%"
        else:
            imp = "N/A"
        print(f"  {key:<22}  {rr_val:>12.2f}  {rl_val:>12.2f}  {imp:>12}")

    rr_throughput = metrics_rr["throughput"]
    rl_throughput = metrics_rl["throughput"]
    if rr_throughput > 0:
        pct = 100.0 * (rl_throughput - rr_throughput) / rr_throughput
        imp = f"{pct:+.1f}%"
    else:
        imp = "N/A"
    print(f"  {'throughput':<22}  {rr_throughput:>12.2f}  {rl_throughput:>12.2f}  {imp:>12}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RL scheduler vs Round Robin on bursty workload"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rl/checkpoints/ppo_scheduler.zip",
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--workload-seed",
        type=int,
        default=42,
        help="Random seed for workload generation",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=40,
        help="Number of tasks in workload",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=2,
        help="Number of CPU cores",
    )
    parser.add_argument(
        "--quantum",
        type=int,
        default=2,
        help="Round Robin time quantum",
    )
    args = parser.parse_args()

    run_comparison(
        model_path=args.model,
        num_cores=args.cores,
        num_tasks=args.tasks,
        workload_seed=args.workload_seed,
        rr_quantum=args.quantum,
    )


if __name__ == "__main__":
    main()
