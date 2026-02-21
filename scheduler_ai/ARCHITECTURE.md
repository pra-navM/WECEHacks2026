# Scheduler AI — Code Architecture Documentation

This document explains how each module in the multicore CPU scheduling simulator works and how the pieces fit together.

---

## High-Level Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  workload/      │     │  simulator/      │     │  main.py         │
│  generator.py   │────▶│  simulation.py   │◀───▶│  (CLI)           │
│                 │     │                  │     │                 │
│  Creates Tasks  │     │  Orchestrates     │     │  Parses args,    │
│  with metadata  │     │  tick loop        │     │  runs sim,       │
│                 │     │                  │     │  prints results   │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
              ┌─────────┐  ┌─────────┐  ┌─────────────────┐
              │ task.py │  │ core.py │  │ scheduler_base   │
              │         │  │         │  │ fcfs, rr, sjf    │
              │ Task    │  │ Core    │  │ (policy logic)   │
              └─────────┘  └─────────┘  └─────────────────┘
```

---

## Package Overview

| Package   | Role |
|-----------|------|
| `simulator/` | Core simulation logic: tasks, cores, schedulers, tick engine |
| `workload/`  | Generates task lists (uniform or mice/elephants) |
| `metrics/`   | Performance aggregation (stub for Phase 2+) |
| `rl/`        | RL environment, agent, training (stub for Phase 3) |

---

## File-by-File Reference

### 1. `simulator/task.py` — Task Model

**Purpose:** Defines the unit of work that gets scheduled and executed.

**Key types:**
- **`TickResult`** — Enum: `RUNNING`, `COMPLETED`, `IO_BLOCKED`. Returned by `run_one_tick()` to tell the simulation what happened.
- **`TaskMetadata`** — Frozen dataclass: `memory_rss`, `cache_locality`, `io_blocking_prob`, `buffer_requirement`. Optional; defaults reproduce Phase 1 behavior.
- **`Task`** — The main class.

**Task attributes:**
- `task_id`, `arrival_time`, `burst_time`, `remaining_time`, `start_time`, `completion_time` — core scheduling fields
- `metadata` — resource profile (Phase 2)
- `last_core_id` — for cache-affinity tracking
- `io_block_remaining` — countdown when task is I/O blocked

**Key methods:**
- `run_one_tick(current_time, rng)` — Executes one time unit; may return `IO_BLOCKED` if I/O wait triggers
- `tick_io_block()` — Advances I/O wait countdown; returns `True` when unblocked
- `is_complete()` — `remaining_time == 0`
- `turnaround_time`, `waiting_time` — Computed properties after completion

**Used by:** `core.py` (calls `run_one_tick`), `workload/generator.py` (creates Tasks), `simulation.py` (manages blocked tasks), all schedulers (receive Tasks)

---

### 2. `simulator/core.py` — CPU Core Model

**Purpose:** Represents a single CPU core that runs one task at a time. Handles context-switch latency and preemption.

**Key attributes:**
- `core_id` — Unique identifier
- `current_task` — Task currently assigned (or `None` if idle)
- `context_switch_cost` — Dead ticks when a new task is assigned
- `context_switch_remaining` — Countdown during switch
- `ticks_on_core` — How long the current task has run (used by RR for preemption)

**Key methods:**
- `assign_task(task)` — Assigns a task; sets `context_switch_remaining` so the core is "dead" for that many ticks
- `tick(current_time, rng)` — Advances one tick: either consumes context-switch time or runs `task.run_one_tick()`. Returns `(task_or_None, TickResult)`.
- `preempt()` — Removes the current task without completing it; used when the scheduler requests preemption
- `is_idle()` — `current_task is None`

**Used by:** `simulation.py` (creates cores, calls `assign_task`, `tick`, `preempt`)

**Depends on:** `task.py` (Task, TickResult)

---

### 3. `simulator/scheduler_base.py` — Scheduler Interface

**Purpose:** Abstract base class defining the contract between the simulation engine and any scheduling policy. The engine never contains policy logic; it only calls these methods.

**Abstract methods (must implement):**
- `add_task(task)` — New task has arrived; add to ready queue
- `get_next_task(current_time)` — Core is idle; return the next task to run (or `None`)
- `on_task_complete(task)` — Task finished; update internal state
- `has_pending_tasks()` — Are there tasks still in the ready queue?

**Concrete hooks (optional overrides):**
- `check_preemption(core_id, task, ticks_on_core, current_time)` — Return `True` to preempt. Default: `False`
- `on_task_blocked(task)` — Task entered I/O wait. Default: no-op
- `on_task_unblocked(task)` — Task finished I/O wait. Default: `add_task(task)`

**Used by:** `simulation.py` (calls all of these), `fcfs.py`, `round_robin.py`, `sjf.py` (implement the interface)

**Depends on:** `task.py` (Task)

---

### 4. `simulator/fcfs.py` — First Come First Served

**Purpose:** Non-preemptive FIFO scheduler.

**Implementation:** Internal `deque`; `add_task` appends, `get_next_task` popleft. Inherits default `check_preemption` (no preemption) and `on_task_blocked`/`on_task_unblocked`.

**Depends on:** `scheduler_base.py`, `task.py`

---

### 5. `simulator/round_robin.py` — Round Robin

**Purpose:** Preemptive scheduler with configurable time quantum.

**Implementation:** Same queue structure as FCFS, but overrides `check_preemption` to return `True` when `ticks_on_core >= time_quantum`. The simulation engine then preempts the task and re-adds it via `add_task()`.

**Depends on:** `scheduler_base.py`, `task.py`

---

### 6. `simulator/sjf.py` — Shortest Job First

**Purpose:** Non-preemptive scheduler that picks the task with smallest `remaining_time`.

**Implementation:** Internal list; `get_next_task` sorts by `(remaining_time, arrival_time, task_id)` and pops the smallest. Inherits default preemption/blocking hooks.

**Depends on:** `scheduler_base.py`, `task.py`

---

### 7. `simulator/simulation.py` — Simulation Engine

**Purpose:** Orchestrates the tick-based simulation. Contains no scheduling policy logic; it only drives the clock and calls the scheduler.

**Tick loop (each time unit):**
1. **Admit arrivals** — Add tasks whose `arrival_time <= current_time` to the scheduler
2. **Unblock I/O** — For each blocked task, call `tick_io_block()`; if unblocked, call `scheduler.on_task_unblocked(task)`
3. **Check preemption** — For each busy core, call `scheduler.check_preemption()`; if `True`, preempt and re-add task
4. **Tick cores** — Call `core.tick()` for each core; handle `COMPLETED` (increment done count) and `IO_BLOCKED` (add to blocked list)
5. **Dispatch** — For each idle core, call `scheduler.get_next_task()` and `core.assign_task()`
6. Increment `current_time`

**Constructor params:** `num_cores`, `scheduler`, `tasks`, `context_switch_cost`, `seed`

**Returns:** The original task list (tasks are mutated in place with `start_time`, `completion_time`, etc.)

**Depends on:** `core.py`, `scheduler_base.py`, `task.py`

---

### 8. `workload/generator.py` — Workload Generation

**Purpose:** Produces deterministic task lists for the simulation.

**Functions:**
- **`generate_workload(num_tasks, seed, ...)`** — Uniform random arrivals and burst times. Creates plain `Task` objects (no metadata). Phase 1 compatible.
- **`generate_workload_mixed(num_tasks, seed, mice_fraction, ...)`** — Mice-and-elephants model: short lightweight tasks vs long resource-heavy tasks. Populates `TaskMetadata` (RSS, cache locality, I/O prob, buffer). Bursty arrivals via random burst centers.

**Depends on:** `task.py` (Task, TaskMetadata)

---

### 9. `main.py` — CLI Entry Point

**Purpose:** Parses command-line arguments, generates workload, runs simulation, prints results.

**Flow:**
1. Parse args: `--cores`, `--scheduler`, `--tasks`, `--seed`, `--quantum`, `--context-switch-cost`, `--workload-type`
2. Call `generate_workload` or `generate_workload_mixed` based on `--workload-type`
3. Instantiate scheduler (FCFS, RR, or SJF)
4. Create `Simulation(cores, scheduler, tasks, context_switch_cost, seed)`
5. Call `sim.run()`
6. Print per-task table and summary (avg turnaround, avg wait)

**Depends on:** `simulator/*`, `workload/generator.py`

---

### 10. `metrics/performance.py` — Metrics (Stub)

**Purpose:** Placeholder for Phase 2+ metric aggregation (throughput, fairness, utilization). Currently raises `NotImplementedError`.

**Intended use:** Accept completed task list from `Simulation.run()`, return `Dict[str, float]` of metric names to values.

---

### 11. `rl/env.py`, `rl/agent.py`, `rl/train.py` — RL Stubs

**Purpose:** Placeholders for Phase 3 RL integration.

- **`env.py`** — Will wrap the simulation as a Gymnasium environment: `step(action)` returns `(obs, reward, done, info)`
- **`agent.py`** — Will implement an RL agent that subclasses `SchedulerBase` and uses a learned policy in `get_next_task`
- **`train.py`** — Will orchestrate episodes and policy updates

---

## Dependency Graph (Simplified)

```
main.py
  ├── workload/generator.py ──▶ task.py
  ├── simulator/simulation.py
  │     ├── core.py ──▶ task.py
  │     ├── scheduler_base.py ──▶ task.py
  │     └── fcfs.py, round_robin.py, sjf.py ──▶ scheduler_base.py, task.py
  └── (prints Task results)

metrics/performance.py ──▶ task.py  (future)
rl/* ──▶ (future: simulation, scheduler_base)
```

---

## Execution Flow (Single Tick)

```
Simulation.run() loop
    │
    ├─▶ 1. Admit: scheduler.add_task(task) for each new arrival
    │
    ├─▶ 2. Unblock: for blocked task, task.tick_io_block()
    │         if unblocked → scheduler.on_task_unblocked(task)
    │
    ├─▶ 3. Preempt: for each busy core,
    │         if scheduler.check_preemption(...) → core.preempt(), scheduler.add_task(preempted)
    │
    ├─▶ 4. Tick: for each core, core.tick(time, rng)
    │         core calls task.run_one_tick(time, rng)
    │         if COMPLETED → scheduler.on_task_complete(task)
    │         if IO_BLOCKED → scheduler.on_task_blocked(task), add to blocked list
    │
    ├─▶ 5. Dispatch: for each idle core,
    │         task = scheduler.get_next_task(time)
    │         if task → core.assign_task(task)
    │
    └─▶ 6. current_time += 1
```

---

## Extensibility for RL

1. **Policy swap** — The simulation only knows `SchedulerBase`. An RL scheduler implements `get_next_task()` with a learned policy; no engine changes.
2. **Stepped execution** — The tick loop can be refactored into `step(action)` for Gymnasium: each step runs one tick, returns observation (queue state, core state) and reward (e.g., negative turnaround).
3. **Determinism** — Seeded RNG for workload and I/O blocking enables reproducible training episodes.
