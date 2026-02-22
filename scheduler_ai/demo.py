import subprocess
import time
import pandas as pd # Optional, for better table formatting if you have it

def run_bench(scheduler, tasks, cost=15):
    print(f"🚀 Running {scheduler.upper()}...")
    # Added --cores 2 to ensure consistency
    cmd = f"python main.py --scheduler {scheduler} --tasks {tasks} --context-switch-cost {cost} --workload-type mixed --cores 2"
    
    start = time.time()
    try:
        result = subprocess.check_output(cmd, shell=True).decode()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {scheduler}: {e}")
        return None, None

    end = time.time()
    wall_time = end - start
    
    wait = 0.0
    for line in result.split('\n'):
        if "Avg Wait" in line:
            # Extract numbers like '12.34' from 'Avg Wait: 12.34'
            wait = float(''.join(c for c in line.split(':')[-1] if c.isdigit() or c == '.'))
            break
            
    return wait, wall_time

# --- Benchmark Configuration ---
num_tasks = 500
schedulers_to_test = ["sjf", "governor", "fast-governor"]
results = {}

print(f"📊 Starting Benchmark: {num_tasks} Tasks, Mixed Workload")
print("-" * 50)

for sched in schedulers_to_test:
    avg_wait, total_time = run_bench(sched, num_tasks)
    if avg_wait is not None:
        results[sched] = {"wait": avg_wait, "runtime": total_time}

# --- Results Presentation ---
print("\n" + "═" * 60)
print(f"🏆 FINAL BENCHMARK RESULTS ({num_tasks} Tasks)")
print(f"{'Scheduler':<15} | {'Avg Wait (Ticks)':<18} | {'Wall Time (s)':<15}")
print("-" * 60)

for name, metrics in results.items():
    print(f"{name.upper():<15} | {metrics['wait']:<18.2f} | {metrics['runtime']:<15.3f}")

print("═" * 60)

# --- Insights ---
if "sjf" in results and "fast-governor" in results:
    efficiency = ((results['sjf']['wait'] - results['fast-governor']['wait']) / results['sjf']['wait']) * 100
    print(f"💡 AI Efficiency Gain over SJF: {efficiency:.2f}%")

if "governor" in results and "fast-governor" in results:
    speedup = results['governor']['runtime'] / results['fast-governor']['runtime']
    print(f"⚡ Quantization Speedup: {speedup:.2f}x faster execution")
    
print("═" * 60)