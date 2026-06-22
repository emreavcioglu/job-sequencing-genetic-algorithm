"""
Ant Colony Optimization for Permutation Flow Shop Scheduling Problem (PFSP)
Solves Just-In-Sequence (JIS) production planning using Swarm Intelligence
"""

import os
import random
import numpy as np
import time
from typing import List, Tuple, Dict


class ACO_PFSP:
    """
    Ant Colony Optimization solver for Permutation Flow Shop Scheduling Problem
    
    Problem: Find optimal job sequence that minimizes total completion time
    Constraints: 
    - Fixed machine order (all jobs visit machines in same sequence)
    - No preemption
    - Each job appears exactly once (permutation)
    - One job per machine at a time
    """
    
    def __init__(self, 
                 time_matrix: np.ndarray,
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 q: float = 100.0,
                 elite_weight: float = 2.0):
        """
        Initialize ACO solver
        
        Args:
            time_matrix: Processing time matrix [n_jobs x n_machines]
            n_ants: Number of ants in colony
            n_iterations: Number of iterations
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            rho: Pheromone evaporation rate (0-1)
            q: Pheromone deposit factor
            elite_weight: Weight for best solution pheromone reinforcement
        """
        self.time_matrix = time_matrix
        self.n_jobs, self.n_machines = time_matrix.shape
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elite_weight = elite_weight
        
        # Initialize pheromone matrix (jobs x jobs) for adjacency pheromones
        self.pheromone = np.ones((self.n_jobs, self.n_jobs))
        
        # Precompute delay matrix and total processing time
        self.delay_matrix, self.total_processing_time = self._precompute_delays()
        
        # Best solution tracking
        self.best_sequence = None
        self.best_makespan = float('inf')
        self.iteration_best_makespans = []
        self.convergence_history = []
        
    def _precompute_delays(self) -> Tuple[np.ndarray, int]:
        """
        Precompute delay matrix for fast makespan calculation
        delay_matrix[i,k] = additional delay when job k follows job i
        """
        delay_matrix = np.zeros((self.n_jobs, self.n_jobs), dtype=int)
        
        for i in range(self.n_jobs):
            for k in range(self.n_jobs):
                if i == k:
                    continue
                
                max_delay = 0
                for j in range(1, self.n_machines + 1):
                    sum_i = np.sum(self.time_matrix[i, :j])
                    sum_k = np.sum(self.time_matrix[k, :j-1])
                    delay = sum_i - sum_k
                    if delay > max_delay:
                        max_delay = delay
                
                delay_matrix[i, k] = max_delay
        
        total_processing_time = np.sum(self.time_matrix)
        return delay_matrix, total_processing_time
    
    def calculate_makespan(self, sequence: List[int]) -> float:
        """
        Calculate average completion time (makespan) for a job sequence
        Uses precomputed delay matrix for efficiency
        """
        n = len(sequence)
        total_delay_term = 0
        
        for i in range(n - 1):
            job_i = sequence[i]
            job_k = sequence[i + 1]
            weight = n - 1 - i
            total_delay_term += weight * self.delay_matrix[job_i, job_k]
        
        return (total_delay_term + self.total_processing_time) / n
    
    def _calculate_heuristic(self) -> np.ndarray:
        """
        Calculate heuristic information matrix
        Heuristic = 1 / (delay + 1) - prefer transitions with low delay
        """
        heuristic = np.zeros((self.n_jobs, self.n_jobs))
        
        for i in range(self.n_jobs):
            for j in range(self.n_jobs):
                if i != j:
                    heuristic[i, j] = 1.0 / (self.delay_matrix[i, j] + 1)
        
        return heuristic
    
    def _select_next_job(self, current_job: int, unvisited: List[int], 
                         heuristic: np.ndarray) -> int:
        """
        Select next job using pheromone and heuristic information
        Probability = (pheromone^alpha) * (heuristic^beta)
        """
        if not unvisited:
            return None
        
        # Calculate selection probabilities
        probabilities = []
        for job in unvisited:
            pheromone_factor = self.pheromone[current_job, job] ** self.alpha
            heuristic_factor = heuristic[current_job, job] ** self.beta
            probabilities.append(pheromone_factor * heuristic_factor)
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            # Random selection if all probabilities are zero
            return random.choice(unvisited)
        
        probabilities = [p / total for p in probabilities]
        
        # Select job based on probabilities
        return np.random.choice(unvisited, p=probabilities)
    
    def _construct_solution(self, heuristic: np.ndarray) -> List[int]:
        """
        Ant constructs a complete solution (job sequence)
        """
        # Start with random job
        sequence = [random.randint(0, self.n_jobs - 1)]
        unvisited = [j for j in range(self.n_jobs) if j != sequence[0]]
        
        # Build sequence step by step
        while unvisited:
            current_job = sequence[-1]
            next_job = self._select_next_job(current_job, unvisited, heuristic)
            sequence.append(next_job)
            unvisited.remove(next_job)
        
        return sequence
    
    def _update_pheromones(self, all_sequences: List[List[int]], 
                          all_makespans: List[float]):
        """
        Update pheromone matrix based on ant solutions
        """
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Deposit pheromones from all ants
        for sequence, makespan in zip(all_sequences, all_makespans):
            # Pheromone deposit inversely proportional to makespan
            deposit = self.q / makespan
            
            for i in range(len(sequence) - 1):
                job_i = sequence[i]
                job_j = sequence[i + 1]
                self.pheromone[job_i, job_j] += deposit
        
        # Elite ant strategy - reinforce best solution
        if self.best_sequence is not None:
            elite_deposit = self.elite_weight * self.q / self.best_makespan
            for i in range(len(self.best_sequence) - 1):
                job_i = self.best_sequence[i]
                job_j = self.best_sequence[i + 1]
                self.pheromone[job_i, job_j] += elite_deposit
    
    def _local_search_2opt(self, sequence: List[int]) -> List[int]:
        """
        Apply 2-opt local search to improve solution
        """
        best_seq = sequence.copy()
        best_makespan = self.calculate_makespan(best_seq)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(best_seq) - 1):
                for j in range(i + 2, len(best_seq)):
                    # Reverse segment [i+1:j]
                    new_seq = best_seq.copy()
                    new_seq[i+1:j+1] = reversed(new_seq[i+1:j+1])
                    new_makespan = self.calculate_makespan(new_seq)
                    
                    if new_makespan < best_makespan:
                        best_seq = new_seq
                        best_makespan = new_makespan
                        improved = True
                        break
                if improved:
                    break
        
        return best_seq
    
    def solve(self, use_local_search: bool = True, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Run ACO algorithm to find optimal job sequence
        
        Returns:
            best_sequence: Optimal job ordering
            best_makespan: Minimum completion time
        """
        heuristic = self._calculate_heuristic()
        
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            # Each ant constructs a solution
            all_sequences = []
            all_makespans = []
            
            for ant in range(self.n_ants):
                # Construct solution
                sequence = self._construct_solution(heuristic)
                
                # Apply local search
                if use_local_search and random.random() < 0.3:  # 30% of ants use local search
                    sequence = self._local_search_2opt(sequence)
                
                # Evaluate solution
                makespan = self.calculate_makespan(sequence)
                
                all_sequences.append(sequence)
                all_makespans.append(makespan)
                
                # Update global best
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_sequence = sequence.copy()
            
            # Update pheromones
            self._update_pheromones(all_sequences, all_makespans)
            
            # Track iteration best
            iter_best = min(all_makespans)
            self.iteration_best_makespans.append(iter_best)
            self.convergence_history.append(self.best_makespan)
            
            if verbose and (iteration + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration + 1}/{self.n_iterations} | "
                      f"Best: {self.best_makespan:.2f} | "
                      f"Iter Best: {iter_best:.2f} | "
                      f"Time: {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ACO Optimization Complete!")
            print(f"{'='*60}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Best Makespan: {self.best_makespan:.2f}")
            print(f"Best Sequence: {[int(j)+1 for j in self.best_sequence]}")  # 1-indexed for display
            print(f"{'='*60}\n")
        
        return self.best_sequence, self.best_makespan
    
    def get_schedule_details(self, sequence: List[int]) -> Dict:
        """
        Get detailed scheduling information for visualization
        Returns Gantt chart data
        """
        schedule = {
            'jobs': [],
            'machines': [],
            'start_times': [],
            'end_times': [],
            'job_completion_times': [0] * self.n_jobs
        }
        
        # Track completion time for each machine
        machine_ready_time = [0] * self.n_machines
        
        # Track completion time for each job
        job_completion_on_machine = {}
        
        for job_idx in sequence:
            job_completion_on_machine[job_idx] = [0] * self.n_machines
            
            for machine in range(self.n_machines):
                # Job can start when both:
                # 1. Machine is ready
                # 2. Job completed on previous machine (if not first machine)
                if machine == 0:
                    start_time = machine_ready_time[machine]
                else:
                    start_time = max(machine_ready_time[machine], 
                                    job_completion_on_machine[job_idx][machine - 1])
                
                processing_time = self.time_matrix[job_idx, machine]
                end_time = start_time + processing_time
                
                # Update times
                machine_ready_time[machine] = end_time
                job_completion_on_machine[job_idx][machine] = end_time
                
                # Store for Gantt chart
                schedule['jobs'].append(job_idx)
                schedule['machines'].append(machine)
                schedule['start_times'].append(start_time)
                schedule['end_times'].append(end_time)
            
            # Final completion time for this job
            schedule['job_completion_times'][job_idx] = job_completion_on_machine[job_idx][-1]
        
        return schedule


def load_problem_data(filepath: str) -> Tuple[np.ndarray, int, int]:
    """
    Load PFSP problem data from file
    
    File format:
        Line 1: number of jobs
        Line 2: number of machines
        Next n lines: processing time matrix (jobs x machines)
    """
    with open(filepath, 'r') as f:
        n_jobs = int(f.readline().strip())
        n_machines = int(f.readline().strip())
        time_matrix = []
        
        for _ in range(n_jobs):
            row = list(map(int, f.readline().strip().split()))
            time_matrix.append(row)
    
    return np.array(time_matrix, dtype=int), n_jobs, n_machines


def main():
    """
    Console-based ACO solver for PFSP
    """
    print("\n" + "="*70)
    print(" ACO for Permutation Flow Shop Scheduling Problem (PFSP)")
    print(" Just-In-Sequence (JIS) Production Planning")
    print(" Swarm Intelligence Approach")
    print("="*70 + "\n")
    
    # Load problem data
    data_file = "datenCustomer_5_3.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        print("Using default small problem...")
        # Default small problem
        time_matrix = np.array([
            [2, 1, 2],
            [1, 1, 1],
            [3, 2, 3],
            [1, 2, 2]
        ])
        n_jobs, n_machines = time_matrix.shape
    else:
        time_matrix, n_jobs, n_machines = load_problem_data(data_file)
    
    print(f"Problem Size: {n_jobs} jobs × {n_machines} machines")
    print(f"\nProcessing Time Matrix:")
    print(f"{'-'*50}")
    print(f"{'Job':<6} | " + " ".join([f"M{i+1:<3}" for i in range(n_machines)]))
    print(f"{'-'*50}")
    for i in range(n_jobs):
        print(f"J{i+1:<5} | " + " ".join([f"{time_matrix[i,j]:<4}" for j in range(n_machines)]))
    print(f"{'-'*50}\n")
    
    # ACO Parameters
    print("ACO Parameters:")
    print(f"  • Number of ants: 20")
    print(f"  • Iterations: 100")
    print(f"  • Alpha (pheromone): 1.0")
    print(f"  • Beta (heuristic): 2.0")
    print(f"  • Evaporation rate: 0.1")
    print(f"  • Local search: Enabled (30% of ants)")
    print()
    
    # Initialize and run ACO
    aco = ACO_PFSP(
        time_matrix=time_matrix,
        n_ants=20,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q=100.0,
        elite_weight=2.0
    )
    
    best_sequence, best_makespan = aco.solve(use_local_search=True, verbose=True)
    
    # Display detailed schedule
    schedule = aco.get_schedule_details(best_sequence)
    
    print("\nDetailed Schedule (Gantt Chart Data):")
    print(f"{'-'*70}")
    print(f"{'Machine':<10} {'Job':<10} {'Start':<10} {'End':<10} {'Duration':<10}")
    print(f"{'-'*70}")
    
    for i in range(len(schedule['jobs'])):
        job = schedule['jobs'][i]
        machine = schedule['machines'][i]
        start = schedule['start_times'][i]
        end = schedule['end_times'][i]
        duration = end - start
        
        print(f"M{machine+1:<9} J{job+1:<9} {start:<10} {end:<10} {duration:<10}")
    
    print(f"{'-'*70}\n")
    
    print("Job Completion Times:")
    for job_idx in best_sequence:
        completion_time = schedule['job_completion_times'][job_idx]
        print(f"  Job {job_idx+1}: {completion_time}")
    
    print(f"\nMakespan (Total Completion Time): {max(schedule['end_times'])}")
    print(f"Average Completion Time: {best_makespan:.2f}")
    print()


if __name__ == "__main__":
    main()
