# CPU Scheduling Simulator

A multicore CPU scheduling simulator that supports multiple scheduling algorithms (FCFS, Round Robin, SJF) and can be extended with RL-based scheduling.

## Quick Start

### Running the Simulation

From the `scheduler_ai` directory:

```bash
python main.py [options]
```

### Example Commands

```bash
# Basic FCFS scheduler with 10 tasks
python main.py --tasks 10 --scheduler fcfs

# Round Robin with custom quantum
python main.py --tasks 10 --scheduler rr --quantum 3

# Shortest Job First
python main.py --tasks 10 --scheduler sjf

# Mixed workload (mice and elephants) with 4 cores
python main.py --tasks 20 --scheduler fcfs --workload-type mixed --cores 4

# With context switch cost
python main.py --tasks 10 --scheduler rr --context-switch-cost 1
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cores` | 2 | Number of CPU cores |
| `--scheduler` | fcfs | Scheduling policy: `fcfs`, `rr`, `sjf` |
| `--tasks` | 10 | Number of tasks to generate |
| `--seed` | 42 | Random seed for reproducibility |
| `--quantum` | 2 | Time quantum for Round Robin scheduler |
| `--context-switch-cost` | 0 | Dead ticks per context switch |
| `--workload-type` | uniform | Workload type: `uniform` or `mixed` |

## Project Structure

```
scheduler_ai/
├── main.py                 # CLI entry point
├── simulator/              # Core simulation engine
│   ├── core.py            # CPU core model
│   ├── task.py            # Task model
│   ├── simulation.py      # Main simulation loop
│   ├── scheduler_base.py  # Abstract scheduler interface
│   ├── fcfs.py            # First Come First Served
│   ├── round_robin.py     # Round Robin scheduler
│   └── sjf.py             # Shortest Job First
├── workload/              # Workload generation
│   └── generator.py      # Task generation utilities
├── metrics/               # Performance metrics (Phase 2)
└── rl/                    # Reinforcement learning (Phase 3)
```

## Requirements

- Python 3.9+ (uses modern type hints: `list[str]`, `dict[str, Type]`)
- Standard library only (no external dependencies)

## Output Format

The simulator prints:
- Per-task statistics: ID, Arrival, Burst, Start, End, Turnaround, Wait
- Summary statistics: Average Turnaround Time, Average Wait Time
- Additional columns for mixed workloads: IO_Prob, RSS (memory)

## Notes for Hackathon

- All import errors have been fixed - the project runs directly with `python main.py`
- The simulation is deterministic (uses seeded random number generators)
- RL and metrics modules are placeholders for future phases
- The codebase is ready for demonstration and extension
