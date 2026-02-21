# CPU Scheduling Simulator

A multicore CPU scheduling simulator that supports multiple scheduling algorithms (FCFS, Round Robin, SJF) and RL-based scheduling using MaskablePPO.

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

# RL scheduler (requires trained model - see Training section)
python main.py --tasks 20 --scheduler rl --workload-type mixed --model rl/checkpoints/ppo_scheduler.zip
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cores` | 2 | Number of CPU cores |
| `--scheduler` | fcfs | Scheduling policy: `fcfs`, `rr`, `sjf`, or `rl` |
| `--model` | rl/checkpoints/ppo_scheduler.zip | Path to trained RL model (only for `--scheduler rl`) |
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
- Standard library only for baseline schedulers (FCFS, RR, SJF)
- For RL scheduler: `gymnasium`, `numpy`, `stable-baselines3`, `sb3-contrib`

Install RL dependencies:
```bash
pip install -r requirements.txt
```

## Output Format

The simulator prints:
- Per-task statistics: ID, Arrival, Burst, Start, End, Turnaround, Wait
- Summary statistics: Average Turnaround Time, Average Wait Time
- Additional columns for mixed workloads: IO_Prob, RSS (memory)

## Training RL Scheduler

Train a MaskablePPO agent on bursty workloads:

```bash
# Train for 200k timesteps (default)
python -m rl.train

# Custom training
python -m rl.train --timesteps 500000 --save rl/checkpoints/my_model.zip --cores 4
```

The trained model will be saved to `rl/checkpoints/ppo_scheduler.zip` by default.

## Comparative Analysis

Compare RL scheduler vs Round Robin on the same workload:

```bash
python -m rl.compare --model rl/checkpoints/ppo_scheduler.zip --tasks 40 --workload-seed 42
```

This runs both schedulers on identical workloads and reports metrics side-by-side.

## Notes for Hackathon

- All import errors have been fixed - the project runs directly with `python main.py`
- The simulation is deterministic (uses seeded random number generators)
- RL scheduler requires a trained model - train first with `python -m rl.train`
- Baseline schedulers (FCFS, RR, SJF) work without any dependencies
- The codebase is ready for demonstration and extension
