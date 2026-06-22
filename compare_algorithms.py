"""
Comparison: Genetic Algorithm vs Ant Colony Optimization
For Permutation Flow Shop Scheduling Problem (PFSP)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add paths
sys.path.append('job-sequencing-genetic-algorithm-main')

# Import both algorithms
from aco_pfsp_solver import ACO_PFSP, load_problem_data
import ga_functions as ga


def run_ga_comparison(time_matrix, n_jobs, n_machines):
    """
    Run Genetic Algorithm on the problem
    """
    print("\n" + "="*70)
    print("GENETIC ALGORITHM")
    print("="*70)
    
    # Precompute delays
    delay_matrix, total_processing_time = ga.precompute_delays(time_matrix)
    
    # GA Parameters
    population_size = 50
    generations = 250
    mutation_rate = 0.1
    
    print(f"\nGA Parameters:")
    print(f"  Population Size: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Mutation Rate: {mutation_rate}")
    print(f"  Crossover: Ordered Crossover (OX)")
    print(f"  Mutation: Insertion + Inversion")
    
    # Initialize population
    population = ga.initialize_population(n_jobs, population_size)
    
    # Track best solutions
    best_solutions_ga = []
    start_time = time.time()
    
    best_individual = None
    best_makespan = float('inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = ga.evaluate_fitness(population, delay_matrix, total_processing_time)
        
        # Track best
        current_best, current_makespan = ga.get_best_solution(
            population, fitness_scores, delay_matrix, total_processing_time
        )
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_individual = current_best.copy()
        
        best_solutions_ga.append(best_makespan)
        
        if (gen + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {gen+1}/{generations} | Best: {best_makespan:.2f} | Time: {elapsed:.2f}s")
        
        # Selection
        selected_parents = [
            ga.tournament_selection(population, fitness_scores, tournament_size=5)
            for _ in range(population_size // 2)
        ]
        
        # Create next generation
        population = ga.create_next_generation(
            selected_parents, population_size, mutation_rate
        )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"GA Complete!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Makespan: {best_makespan:.2f}")
    print(f"Best Sequence: {[int(j)+1 for j in best_individual]}")
    print(f"{'='*60}\n")
    
    return best_individual, best_makespan, best_solutions_ga, total_time


def run_aco_comparison(time_matrix, n_jobs, n_machines):
    """
    Run Ant Colony Optimization on the problem
    """
    print("\n" + "="*70)
    print("ANT COLONY OPTIMIZATION")
    print("="*70)
    
    # ACO Parameters
    n_ants = 20
    n_iterations = 5
    
    print(f"\nACO Parameters:")
    print(f"  Number of Ants: {n_ants}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Alpha (pheromone): 1.0")
    print(f"  Beta (heuristic): 2.0")
    print(f"  Evaporation: 0.1")
    print(f"  Local Search: Enabled")
    
    # Initialize ACO
    aco = ACO_PFSP(
        time_matrix=time_matrix,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q=100.0,
        elite_weight=2.0
    )
    
    start_time = time.time()
    # Run optimization
    best_sequence, best_makespan = aco.solve(use_local_search=True, verbose=True)
    aco_total_time = time.time() - start_time
    return best_sequence, best_makespan, aco.convergence_history, aco_total_time


def plot_comparison(ga_convergence, aco_convergence, ga_time, aco_time, ga_best, aco_best):
    """
    Plot comparison of both algorithms
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence comparison
    ax1 = axes[0]
    ga_iterations = range(1, len(ga_convergence) + 1)
    aco_iterations = range(1, len(aco_convergence) + 1)
    
    ax1.plot(ga_iterations, ga_convergence, 'b-', linewidth=2, label=f'GA (Best: {ga_best:.2f})', alpha=0.7)
    ax1.plot(aco_iterations, aco_convergence, 'r-', linewidth=2, label=f'ACO (Best: {aco_best:.2f})', alpha=0.7)
    
    ax1.set_xlabel('Iteration / Generation', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Makespan (Lower is Better)', fontsize=11, fontweight='bold')
    ax1.set_title('Convergence Comparison: GA vs ACO', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Bar chart comparison
    ax2 = axes[1]
    algorithms = ['Genetic\nAlgorithm', 'Ant Colony\nOptimization']
    makespans = [ga_best, aco_best]
    times = [ga_time, aco_time]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, makespans, width, label='Best Makespan', color=['#3498db', '#e74c3c'], alpha=0.8)
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, times, width, label='Execution Time (s)', color=['#2ecc71', '#f39c12'], alpha=0.8)
    
    ax2.set_ylabel('Makespan', fontsize=11, fontweight='bold', color='#2c3e50')
    ax2_twin.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold', color='#2c3e50')
    ax2.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('comparison_ga_vs_aco.png', dpi=300, bbox_inches='tight')
    print("\n✅ Comparison plot saved as 'comparison_ga_vs_aco.png'")
    plt.show()


def main():
    """
    Main comparison runner
    """
    print("\n" + "="*80)
    print(" ALGORITHM COMPARISON: Genetic Algorithm vs Ant Colony Optimization")
    print(" Permutation Flow Shop Scheduling Problem (PFSP)")
    print("="*80)
    
    # Load problem data
    data_file = "job-sequencing-genetic-algorithm-main/data/customer_50_1.txt"
    
    if os.path.exists(data_file):
        time_matrix, n_jobs, n_machines = load_problem_data(data_file)
        print(f"\n✅ Loaded: {data_file}")
    else:
        print(f"\n⚠️  Data file not found. Using default 4-job problem.")
        time_matrix = np.array([
            [2, 1, 2],
            [1, 1, 1],
            [3, 2, 3],
            [1, 2, 2]
        ])
        n_jobs, n_machines = time_matrix.shape
    
    print(f"\nProblem Size: {n_jobs} jobs × {n_machines} machines\n")
    print("Processing Time Matrix:")
    print("-" * 50)
    print(f"{'Job':<6} | " + " ".join([f"M{i+1:<3}" for i in range(n_machines)]))
    print("-" * 50)
    for i in range(n_jobs):
        print(f"J{i+1:<5} | " + " ".join([f"{time_matrix[i,j]:<4}" for j in range(n_machines)]))
    print("-" * 50)
    
    # Run GA
    ga_sequence, ga_makespan, ga_convergence, ga_time = run_ga_comparison(
        time_matrix, n_jobs, n_machines
    )
    
    # Run ACO
    aco_sequence, aco_makespan, aco_convergence, aco_time = run_aco_comparison(
        time_matrix, n_jobs, n_machines
    )
    
    # Summary comparison
    print("\n" + "="*80)
    print(" FINAL COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Algorithm':<30} {'Best Makespan':<20} {'Time (s)':<15}")
    print("-" * 80)
    print(f"{'Genetic Algorithm':<30} {ga_makespan:<20.2f} {ga_time:<15.2f}")
    print(f"{'Ant Colony Optimization':<30} {aco_makespan:<20.2f} {aco_time:<15.2f}")
    print("-" * 80)
    
    # Winner
    if ga_makespan < aco_makespan:
        improvement = ((aco_makespan - ga_makespan) / aco_makespan) * 100
        print(f"\n🏆 Winner: Genetic Algorithm (Better by {improvement:.2f}%)")
    elif aco_makespan < ga_makespan:
        improvement = ((ga_makespan - aco_makespan) / ga_makespan) * 100
        print(f"\n🏆 Winner: Ant Colony Optimization (Better by {improvement:.2f}%)")
    else:
        print(f"\n🤝 Tie: Both algorithms found the same solution!")
    
    print("\nGA Best Sequence:  " + str([int(j)+1 for j in ga_sequence]))
    print("ACO Best Sequence: " + str([int(j)+1 for j in aco_sequence]))
    
    print("\n" + "="*80)
    
    # Plot comparison
    plot_comparison(ga_convergence, aco_convergence, ga_time, aco_time, ga_makespan, aco_makespan)


if __name__ == "__main__":
    main()
