"""Workload generation for the CPU scheduling simulator."""

from __future__ import annotations

import random
from typing import List

from simulator.task import Task, TaskMetadata


def generate_workload(
    num_tasks: int,
    seed: int = 42,
    arrival_time_range: tuple[int, int] = (0, 20),
    burst_time_range: tuple[int, int] = (1, 10),
) -> List[Task]:
    """Generate a reproducible list of tasks with random parameters.

    Uses a local Random instance seeded with *seed* so that results are
    fully deterministic regardless of external random state.

    Args:
        num_tasks: Number of tasks to generate.
        seed: RNG seed for reproducibility.
        arrival_time_range: Inclusive (min, max) range for arrival times.
        burst_time_range: Inclusive (min, max) range for burst times.

    Returns:
        A list of Task objects sorted by arrival_time.
    """
    rng = random.Random(seed)
    tasks: List[Task] = []

    for i in range(num_tasks):
        arrival = rng.randint(*arrival_time_range)
        burst = rng.randint(*burst_time_range)
        tasks.append(Task(task_id=i, arrival_time=arrival, burst_time=burst))

    tasks.sort(key=lambda t: (t.arrival_time, t.task_id))
    return tasks


def generate_workload_mixed(
    num_tasks: int,
    seed: int = 42,
    mice_fraction: float = 0.8,
    arrival_span: int = 50,
    num_burst_centers: int = 5,
) -> List[Task]:
    """Generate a Mice-and-Elephants workload with bursty arrivals.

    *Mice* are short-lived, lightweight tasks (like web requests).
    *Elephants* are long-running, resource-heavy tasks (like batch jobs).
    Arrivals are clustered around randomly chosen burst centers to
    simulate realistic traffic patterns.

    Args:
        num_tasks: Total number of tasks to generate.
        seed: RNG seed for full reproducibility.
        mice_fraction: Fraction of tasks that are mice (0.0-1.0).
        arrival_span: Time window over which tasks arrive.
        num_burst_centers: Number of arrival burst centers.

    Returns:
        A list of Task objects sorted by arrival_time, each carrying
        populated TaskMetadata.
    """
    rng = random.Random(seed)

    burst_centers = sorted(rng.randint(0, arrival_span) for _ in range(num_burst_centers))
    num_mice = int(num_tasks * mice_fraction)

    tasks: List[Task] = []

    for i in range(num_tasks):
        is_mouse = i < num_mice
        center = rng.choice(burst_centers)
        jitter = rng.randint(-3, 3)
        arrival = max(0, center + jitter)

        if is_mouse:
            burst = rng.randint(1, 5)
            metadata = TaskMetadata(
                memory_rss=rng.randint(1, 4),
                cache_locality=round(rng.uniform(0.0, 0.3), 2),
                io_blocking_prob=round(rng.uniform(0.0, 0.05), 3),
                buffer_requirement=rng.randint(0, 1),
            )
        else:
            burst = rng.randint(20, 100)
            metadata = TaskMetadata(
                memory_rss=rng.randint(8, 32),
                cache_locality=round(rng.uniform(0.3, 0.9), 2),
                io_blocking_prob=round(rng.uniform(0.05, 0.20), 3),
                buffer_requirement=rng.randint(2, 8),
            )

        tasks.append(
            Task(task_id=i, arrival_time=arrival, burst_time=burst, metadata=metadata)
        )

    tasks.sort(key=lambda t: (t.arrival_time, t.task_id))
    return tasks
