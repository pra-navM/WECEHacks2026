"""CLI entry point for the multicore CPU scheduling simulator."""

from __future__ import annotations

import argparse
from typing import Dict, List, Type

from scheduler_ai.simulator.fcfs import FCFSScheduler
from scheduler_ai.simulator.round_robin import RoundRobinScheduler
from scheduler_ai.simulator.scheduler_base import SchedulerBase
from scheduler_ai.simulator.simulation import Simulation
from scheduler_ai.simulator.sjf import SJFScheduler
from scheduler_ai.simulator.task import Task
from scheduler_ai.workload.generator import generate_workload, generate_workload_mixed

SCHEDULERS: Dict[str, Type[SchedulerBase]] = {
    "fcfs": FCFSScheduler,
    "rr": RoundRobinScheduler,
    "sjf": SJFScheduler,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Multicore CPU Scheduling Simulator",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=2,
        help="Number of CPU cores (default: 2)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=list(SCHEDULERS.keys()),
        default="fcfs",
        help="Scheduling policy (default: fcfs)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Number of tasks to generate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for workload generation (default: 42)",
    )
    parser.add_argument(
        "--quantum",
        type=int,
        default=2,
        help="Time quantum for Round Robin (default: 2)",
    )
    parser.add_argument(
        "--context-switch-cost",
        type=int,
        default=0,
        help="Dead ticks per context switch (default: 0)",
    )
    parser.add_argument(
        "--workload-type",
        type=str,
        choices=["uniform", "mixed"],
        default="uniform",
        help="Workload distribution: uniform (Phase 1) or mixed mice/elephants (default: uniform)",
    )
    return parser


def make_scheduler(name: str, quantum: int) -> SchedulerBase:
    """Instantiate the requested scheduler."""
    if name == "rr":
        return RoundRobinScheduler(time_quantum=quantum)
    return SCHEDULERS[name]()


def print_results(
    tasks: List[Task],
    scheduler_name: str,
    num_cores: int,
    context_switch_cost: int,
) -> None:
    """Print per-task results and summary statistics to stdout."""
    has_io = any(t.metadata.io_blocking_prob > 0.0 for t in tasks)

    cols = f"{'ID':>4}  {'Arrival':>7}  {'Burst':>5}  {'Start':>5}  {'End':>5}  {'Turnaround':>10}  {'Wait':>5}"
    if has_io:
        cols += f"  {'IO_Prob':>7}  {'RSS':>4}"
    header = cols
    separator = "-" * len(header)

    label = scheduler_name.upper()
    extras: list[str] = [f"{num_cores} core(s)"]
    if context_switch_cost > 0:
        extras.append(f"cs_cost={context_switch_cost}")
    print(f"\n=== Simulation Results: {label} | {', '.join(extras)} ===\n")
    print(header)
    print(separator)

    tasks_sorted = sorted(tasks, key=lambda t: t.task_id)
    total_turnaround = 0
    total_wait = 0

    for t in tasks_sorted:
        ta = t.turnaround_time if t.turnaround_time is not None else 0
        wa = t.waiting_time if t.waiting_time is not None else 0
        total_turnaround += ta
        total_wait += wa
        line = (
            f"{t.task_id:>4}  {t.arrival_time:>7}  {t.burst_time:>5}  "
            f"{t.start_time:>5}  {t.completion_time:>5}  {ta:>10}  {wa:>5}"
        )
        if has_io:
            line += f"  {t.metadata.io_blocking_prob:>7.3f}  {t.metadata.memory_rss:>4}"
        print(line)

    n = len(tasks_sorted)
    print(separator)
    print(f"  Avg Turnaround: {total_turnaround / n:.2f}")
    print(f"  Avg Wait:       {total_wait / n:.2f}")
    print()


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run simulation, print results."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.workload_type == "mixed":
        workload = generate_workload_mixed(num_tasks=args.tasks, seed=args.seed)
    else:
        workload = generate_workload(num_tasks=args.tasks, seed=args.seed)

    scheduler = make_scheduler(args.scheduler, args.quantum)

    sim = Simulation(
        num_cores=args.cores,
        scheduler=scheduler,
        tasks=workload,
        context_switch_cost=args.context_switch_cost,
        seed=args.seed,
    )

    completed = sim.run()
    print_results(completed, args.scheduler, args.cores, args.context_switch_cost)


if __name__ == "__main__":
    main()
