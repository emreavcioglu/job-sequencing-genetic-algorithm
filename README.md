# Ant Colony Optimization for Permutation Flow Shop Scheduling Problem (PFSP)

## 🎯 Project Overview

This project implements a **Swarm Intelligence** approach to solve the **Permutation Flow Shop Scheduling Problem (PFSP)** used in **Just-In-Sequence (JIS) Production Planning**. The solution uses **Ant Colony Optimization (ACO)** as an alternative to Genetic Algorithms for optimizing job sequences in manufacturing environments.

### Problem Type

**Permutation Flow Shop Scheduling Problem (PFSP)**

- **Domain**: Manufacturing, Production Planning, Industrial Engineering
- **Application**: Automotive assembly, sequential processing systems, JIS production
- **Complexity**: NP-hard combinatorial optimization problem

---

## 📋 Problem Definition

### What We're Solving

Given:
- **N jobs** (products/orders) 
- **M machines** (workstations/stages)
- **Processing time matrix** t(i,j) where each job requires specific time on each machine

Find:
- **Optimal job sequence** that minimizes total completion time (makespan)

### Key Constraints

1. ✅ **Fixed Machine Order**: All jobs follow the same machine sequence (M₁ → M₂ → M₃)
2. ✅ **No Preemption**: Once processing starts, it cannot be interrupted
3. ✅ **Permutation Constraint**: Same job order maintained across all machines
4. ✅ **Single Job per Machine**: No parallel processing on same machine
5. ✅ **Each Job Appears Once**: Valid permutation (no repetition)

### Example Processing Time Matrix

```
        Machine 1   Machine 2   Machine 3
Job 1       2           1           2
Job 2       1           1           1  
Job 3       3           2           3
Job 4       1           2           2
```

### Objective Function

Minimize: **Average Completion Time / Makespan**

```
Objective = (Σ weighted_delays + total_processing_time) / n_jobs
```

---

## 🐜 Why Ant Colony Optimization?

ACO is particularly well-suited for PFSP because:

1. **Natural Fit**: The problem is about discovering good job adjacencies
2. **Pheromone Learning**: Learns which job transitions produce low delays
3. **Exploration-Exploitation**: Balances between trying new sequences and exploiting good ones
4. **Parallel Search**: Multiple ants explore solution space simultaneously
5. **Adaptive**: Pheromone updates guide search toward better solutions

### ACO Algorithm Flow

```
1. Initialize pheromone matrix
2. For each iteration:
   a. Each ant constructs a job sequence
      - Start with random job
      - Select next job based on pheromone + heuristic
      - Continue until complete sequence
   b. Apply local search (optional)
   c. Evaluate all solutions
   d. Update pheromones:
      - Evaporation
      - Deposit based on solution quality
      - Elite reinforcement for best solution
3. Return best sequence found
```

---

## 🚀 Features

### Console Version (`aco_pfsp_solver.py`)

- ✅ Pure Python implementation
- ✅ Efficient delay matrix precomputation
- ✅ Configurable ACO parameters
- ✅ 2-opt local search
- ✅ Elite ant strategy
- ✅ Detailed scheduling output
- ✅ Gantt chart data generation

### GUI Version (`aco_pfsp_gui.py`)

- ✅ Modern interactive interface
- ✅ Real-time convergence visualization
- ✅ Gantt chart display
- ✅ Parameter configuration panel
- ✅ Progress tracking
- ✅ Load custom data files
- ✅ Multi-threaded execution
- ✅ Comprehensive results display

---

## 📦 Installation

### Requirements

```bash
Python 3.7+
numpy
matplotlib
tkinter (usually included with Python)
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy matplotlib
```

---

## 💻 Usage

### Console Version

Run the console-based solver:

```bash
python aco_pfsp_solver.py
```

**Output:**
- Processing time matrix
- ACO parameters
- Iteration progress
- Best sequence found
- Detailed schedule (Gantt data)
- Job completion times

### GUI Version

Launch the graphical interface:

```bash
python aco_pfsp_gui.py
```

**Features:**
1. **Control Panel** (Left):
   - Problem information display
   - Load data files
   - Configure ACO parameters
   - Start/Stop optimization
   - View results

2. **Visualization Panel** (Right):
   - **Gantt Chart**: Visual production schedule
   - **Convergence Plot**: Algorithm progress over time

---

## 🎛️ ACO Parameters

| Parameter | Symbol | Description | Default | Range |
|-----------|--------|-------------|---------|-------|
| **Number of Ants** | - | Colony size | 20 | 10-50 |
| **Iterations** | - | Number of cycles | 100 | 50-500 |
| **Alpha (α)** | α | Pheromone importance | 1.0 | 0.5-2.0 |
| **Beta (β)** | β | Heuristic importance | 2.0 | 1.0-5.0 |
| **Evaporation (ρ)** | ρ | Pheromone decay rate | 0.1 | 0.05-0.3 |
| **Q** | Q | Pheromone deposit factor | 100 | 50-200 |
| **Elite Weight** | - | Best solution reinforcement | 2.0 | 1.0-5.0 |

### Parameter Tuning Tips

- **More Ants**: Better exploration but slower iterations
- **Higher Alpha**: Trust pheromone trails more (exploitation)
- **Higher Beta**: Trust heuristic more (greedy behavior)
- **Lower Evaporation**: Longer pheromone memory
- **Local Search**: Improves quality but increases computation time

---

## 📊 Data File Format

Input files should follow this structure:

```
5                    # Number of jobs
3                    # Number of machines
49 24 35            # Job 1: time on M1, M2, M3
15 44 1             # Job 2: time on M1, M2, M3
47 9 18             # Job 3: time on M1, M2, M3
43 16 2             # Job 4: time on M1, M2, M3
16 10 31            # Job 5: time on M1, M2, M3
```

Sample data files included in `data/` directory.

---

## 📈 Understanding Results

### Best Sequence
```
[3, 4, 1, 2]
```
Means: Process Job 3 first, then Job 4, then Job 1, then Job 2

### Gantt Chart

Shows when each job is processed on each machine:

```
M1: [J3][J4][J1][J2]
M2:   [J3][J4][J1][J2]
M3:     [J3][J4][J1][J2]
```

- Horizontal bars show processing duration
- Gaps indicate machine idle time
- Red line marks total makespan

### Convergence Plot

- **Y-axis**: Solution quality (lower is better)
- **X-axis**: Iteration number
- Shows improvement over time
- Star marks best solution found

---

## 🔬 Algorithm Components

### 1. Pheromone Matrix
```python
pheromone[i][j] = attraction for job j following job i
```

### 2. Heuristic Information
```python
heuristic[i][j] = 1 / (delay[i][j] + 1)
```
Prefer transitions with lower delay.

### 3. Probability Selection
```python
P(job_j) ∝ (pheromone[i][j])^α × (heuristic[i][j])^β
```

### 4. Pheromone Update
```python
# Evaporation
pheromone *= (1 - ρ)

# Deposit
pheromone[i][j] += Q / makespan

# Elite reinforcement
pheromone_best[i][j] += elite_weight × Q / best_makespan
```

### 5. Local Search (2-opt)
Improves solutions by reversing subsequences:
```python
[1,2,3,4,5] → [1,4,3,2,5]  # Reverse segment [2,3,4]
```

---

## 🆚 Comparison: ACO vs Genetic Algorithm

| Aspect | ACO | Genetic Algorithm |
|--------|-----|-------------------|
| **Inspiration** | Ant foraging behavior | Natural evolution |
| **Population** | Solutions = ant paths | Solutions = chromosomes |
| **Memory** | Pheromone matrix | No global memory |
| **Exploration** | Pheromone + heuristic | Crossover + mutation |
| **Exploitation** | Elite ant strategy | Selection pressure |
| **Best For** | Sequential decisions, TSP-like | General permutation problems |
| **Convergence** | Gradual, guided | Can be faster |

**Both are effective for PFSP!** ACO excels at learning good job adjacencies through pheromone trails.

---

## 📝 Code Structure

```
aco_pfsp_solver.py          # Core ACO algorithm + console interface
├── ACO_PFSP class          # Main solver
│   ├── __init__()          # Initialize parameters
│   ├── _precompute_delays() # Build delay matrix
│   ├── calculate_makespan() # Evaluate solution
│   ├── _construct_solution() # Ant builds sequence
│   ├── _update_pheromones() # Learning mechanism
│   ├── _local_search_2opt() # Solution improvement
│   └── solve()             # Main optimization loop
├── load_problem_data()     # File I/O
└── main()                  # Console application

aco_pfsp_gui.py             # GUI application
├── ACO_PFSP_GUI class      # Main window
│   ├── _setup_ui()         # Build interface
│   ├── _run_aco()          # Start optimization
│   ├── _aco_worker()       # Background thread
│   ├── _draw_gantt_chart() # Visualization
│   └── _draw_convergence_plot() # Progress plot
└── main()                  # GUI launcher
```

---

## 🎓 For Presentation

### Key Points to Highlight

1. **Problem Relevance**: Real-world JIS production planning
2. **Algorithm Choice**: Why ACO is well-suited for PFSP
3. **Swarm Intelligence**: Emergent behavior from simple rules
4. **Visualization**: Clear Gantt charts and convergence plots
5. **Performance**: Quality of solutions found
6. **Comparison**: How it relates to GA approach

### Demonstration Flow

1. Show problem definition with example matrix
2. Explain ACO algorithm with ant behavior analogy
3. Run GUI with live visualization
4. Discuss results and schedule interpretation
5. Compare convergence with GA results (if available)

### Questions to Prepare For

- Why use ACO instead of other algorithms?
- How does pheromone update work?
- What is the time complexity?
- Can it handle larger problems?
- How to tune parameters?

---

## 🔍 Example Run

```
ACO for Permutation Flow Shop Scheduling Problem (PFSP)
Just-In-Sequence (JIS) Production Planning
Swarm Intelligence Approach
======================================================================

Problem Size: 5 jobs × 3 machines

Processing Time Matrix:
--------------------------------------------------
Job    | M1   M2   M3  
--------------------------------------------------
J1     | 49   24   35  
J2     | 15   44   1   
J3     | 47   9    18  
J4     | 43   16   2   
J5     | 16   10   31  
--------------------------------------------------

ACO Parameters:
  • Number of ants: 20
  • Iterations: 100
  • Alpha (pheromone): 1.0
  • Beta (heuristic): 2.0
  • Evaporation rate: 0.1
  • Local search: Enabled (30% of ants)

Iteration 10/100 | Best: 145.20 | Iter Best: 147.80 | Time: 0.53s
Iteration 20/100 | Best: 142.60 | Iter Best: 144.40 | Time: 1.08s
...
Iteration 100/100 | Best: 138.40 | Iter Best: 139.20 | Time: 5.24s

============================================================
ACO Optimization Complete!
============================================================
Total Time: 5.24s
Best Makespan: 138.40
Best Sequence: [2, 4, 3, 1, 5]
============================================================
```

---

## 📚 References

### Permutation Flow Shop Scheduling
- Taillard, E. (1993). "Benchmarks for basic scheduling problems"
- Ruiz, R., & Maroto, C. (2005). "A comprehensive review and evaluation of permutation flowshop heuristics"

### Ant Colony Optimization
- Dorigo, M., & Stützle, T. (2004). "Ant Colony Optimization"
- Dorigo, M., et al. (1996). "Ant system: optimization by a colony of cooperating agents"

### ACO for Scheduling
- Stützle, T. (1998). "An ant approach to the flow shop problem"
- Rajendran, C., & Ziegler, H. (2004). "Ant-colony algorithms for permutation flowshop scheduling"

---

## 🤝 Contributing

Improvements welcome! Areas for enhancement:

- Additional local search operators
- Adaptive parameter tuning
- Multi-objective optimization
- Parallel ACO implementation
- Benchmark problem testing
- Alternative pheromone models

---

## 📄 License

Educational project for academic purposes.

---

## 👨‍💻 Author

Created as part of optimization algorithms course project.

**Technology Stack:**
- Python 3.x
- NumPy (numerical computation)
- Matplotlib (visualization)
- Tkinter (GUI framework)

---

## 🎯 Project Goals Achieved

✅ Implement ACO for PFSP  
✅ Console-based solver  
✅ GUI with real-time visualization  
✅ Gantt chart display  
✅ Convergence tracking  
✅ Parameter configuration  
✅ Local search integration  
✅ Comprehensive documentation  

**Ready for presentation! 🎉**
