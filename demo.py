"""
Quick Demo - ACO for PFSP
Run this for a fast demonstration
"""

import numpy as np
from aco_pfsp_solver import ACO_PFSP

print("\n" + "="*70)
print(" QUICK DEMO: Ant Colony Optimization for PFSP")
print("="*70)

# Small test problem from the documentation
time_matrix = np.array([
    [2, 1, 2],  # Job 1: 2 time units on M1, 1 on M2, 2 on M3
    [1, 1, 1],  # Job 2: 1 time unit on each machine
    [3, 2, 3],  # Job 3: 3 time units on M1, 2 on M2, 3 on M3
    [1, 2, 2]   # Job 4: 1 time unit on M1, 2 on M2, 2 on M3
])

n_jobs, n_machines = time_matrix.shape

print(f"\nTest Problem: {n_jobs} jobs × {n_machines} machines")
print("\nProcessing Time Matrix:")
print("  Job 1: [2, 1, 2]")
print("  Job 2: [1, 1, 1]")
print("  Job 3: [3, 2, 3]")
print("  Job 4: [1, 2, 2]")

print("\nRunning ACO with default parameters...")
print("(This will take about 5-10 seconds)")

# Initialize ACO with small iteration count for quick demo
aco = ACO_PFSP(
    time_matrix=time_matrix,
    n_ants=15,           # Fewer ants for speed
    n_iterations=50,     # Fewer iterations for speed
    alpha=1.0,
    beta=2.0,
    rho=0.1
)

# Run optimization
best_sequence, best_makespan = aco.solve(use_local_search=True, verbose=False)

print("\n" + "="*70)
print(" RESULTS")
print("="*70)
print(f"\nBest Sequence Found: {[int(j)+1 for j in best_sequence]}")
print(f"Best Makespan: {best_makespan:.2f}")

# Get detailed schedule
schedule = aco.get_schedule_details(best_sequence)

print("\nProduction Schedule (Gantt Chart):")
print("-" * 70)
print(f"{'Machine':<10} {'Job':<10} {'Start':<10} {'End':<10} {'Duration':<10}")
print("-" * 70)

for i in range(len(schedule['jobs'])):
    job = schedule['jobs'][i]
    machine = schedule['machines'][i]
    start = schedule['start_times'][i]
    end = schedule['end_times'][i]
    duration = end - start
    print(f"M{machine+1:<9} J{job+1:<9} {start:<10} {end:<10} {duration:<10}")

print("-" * 70)

print(f"\nTotal Makespan (Completion Time): {max(schedule['end_times'])}")
print(f"Average Completion Time: {best_makespan:.2f}")

print("\n" + "="*70)
print(" ✅ Demo Complete!")
print("="*70)
print("\nNext steps:")
print("  • Run 'python aco_pfsp_solver.py' for full console version")
print("  • Run 'python aco_pfsp_gui.py' for graphical interface")
print("  • Run 'python compare_algorithms.py' to compare GA vs ACO")
print("\nSee QUICKSTART.md for more information!")
print()
