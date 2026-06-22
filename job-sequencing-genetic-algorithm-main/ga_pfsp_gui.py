"""
GUI Application for Genetic Algorithm - PFSP

Save this script in the same directory as your `ga_functions.py`.
"""

import os
import sys
import threading
import time
import random

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Ensure local module import works regardless of working directory
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ga_functions import (
    precompute_delays,
    initialize_population,
    evaluate_fitness,
    tournament_selection,
    create_next_generation,
    micro_local_search,
    heavy_local_search,
    get_best_solution,
    dynamic_optimizer,
)


def load_problem_data(filepath: str):
    with open(filepath, 'r') as f:
        n_jobs = int(f.readline().strip())
        n_machines = int(f.readline().strip())
        time_matrix = []
        for _ in range(n_jobs):
            row = list(map(int, f.readline().strip().split()))
            time_matrix.append(row)

    return np.array(time_matrix, dtype=int), n_jobs, n_machines


class GA_PFSP_Solver:
    def __init__(self, time_matrix, population_size=50, mutation_rate=0.1):
        self.time_matrix = np.array(time_matrix, dtype=int)
        self.n_jobs, self.n_machines = self.time_matrix.shape
        self.population_size = population_size
        self.mutation_rate = mutation_rate

        self.delayMatrix, self.total_processing_time = precompute_delays(self.time_matrix)
        self.population = initialize_population(self.n_jobs, self.population_size)
        
        self.convergence_history = []
        self.best_sequence = None
        self.best_score = float('inf')  # Corresponds to Average Completion Time

    def get_schedule_details(self, sequence):
        seq = list(sequence)
        n = self.n_jobs
        m = self.n_machines

        start_times = []
        end_times = []
        jobs = []
        machines = []

        completion = np.zeros((n, m), dtype=int)
        job_to_pos = {job: pos for pos, job in enumerate(seq)}

        for idx, job in enumerate(seq):
            for machine in range(m):
                ptime = int(self.time_matrix[job, machine])
                if idx == 0 and machine == 0:
                    start = 0
                elif idx == 0:
                    start = completion[job_to_pos[seq[idx]], machine - 1]
                elif machine == 0:
                    prev_job = seq[idx - 1]
                    start = completion[job_to_pos[prev_job], machine]
                else:
                    prev_job = seq[idx - 1]
                    start = max(
                        completion[job_to_pos[prev_job], machine],
                        completion[job_to_pos[seq[idx]], machine - 1]
                    )

                end = start + ptime
                completion[job_to_pos[seq[idx]], machine] = end

                start_times.append(start)
                end_times.append(end)
                jobs.append(job)
                machines.append(machine)

        schedule = {
            'jobs': jobs,
            'machines': machines,
            'start_times': start_times,
            'end_times': end_times,
            'job_completion_times': {job: int(completion[job_to_pos[job], -1]) for job in seq},
            'makespan': int(np.max(completion))
        }

        return schedule

    def solve(self, n_iterations=100, tournament_size=4, use_micro_ls=False, micro_interval=100, use_heavy_ls=False, callback=None,
              extinction_interval=0, dynamic_mutation=False, base_mutation_rate=None, max_mutation_rate=None, elitism=True):
        
        current_rate = float(base_mutation_rate) if base_mutation_rate is not None else float(self.mutation_rate)
        base_rate = float(base_mutation_rate) if base_mutation_rate is not None else float(self.mutation_rate)
        max_rate = float(max_mutation_rate) if max_mutation_rate is not None else float(self.mutation_rate)

        # Safety check for tournament size vs population size
        safe_tournament_size = min(tournament_size, self.population_size)

        stagnation_counter = 0

        for iteration in range(n_iterations):
            # Evaluate fitness
            fitness_scores = evaluate_fitness(self.population, self.delayMatrix, self.total_processing_time)
            
            # Get best solution in current generation
            gen_best, gen_score = get_best_solution(self.population, fitness_scores, self.delayMatrix, self.total_processing_time)
            
            # Update global best
            if gen_score < self.best_score:
                self.best_score = gen_score
                self.best_sequence = np.array(gen_best).copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Periodic Micro Local Search
            if use_micro_ls and micro_interval > 0 and (iteration > 0 and iteration % micro_interval == 0):
                jobs_to_check = min(10, self.n_jobs) 
                improved_seq, improved_score = micro_local_search(self.best_sequence, jobs_to_check, self.delayMatrix, self.total_processing_time)
                if improved_score < self.best_score:
                    # Lamarckian learning: inject back into population replacing the weakest link
                    worst_idx = np.argmin(fitness_scores)
                    self.population[worst_idx] = improved_seq
                    fitness_scores[worst_idx] = -improved_score
                    
                    self.best_score = improved_score
                    self.best_sequence = improved_seq.copy()

            # Dynamic mutation rate update
            if dynamic_mutation:
                current_rate = dynamic_optimizer(stagnation_counter, current_rate, base_rate, max_rate)

            # Extinction event
            if extinction_interval and extinction_interval > 0 and iteration > 0 and (iteration + 1) % extinction_interval == 0:
                elite_count = max(1, self.population_size // 10)
                sorted_idx = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
                elites = [self.population[i].copy() for i in sorted_idx[:elite_count]]
                
                new_pop = elites.copy()
                new_pop += initialize_population(self.n_jobs, self.population_size - len(new_pop))
                self.population = new_pop
                fitness_scores = evaluate_fitness(self.population, self.delayMatrix, self.total_processing_time)
                stagnation_counter = 0
                current_rate = base_rate

            self.convergence_history.append(self.best_score)

            if callback:
                callback(iteration + 1, n_iterations, self.best_score)

            # Selection & Next Generation
            selected_parents = []
            for _ in range(self.population_size):
                parent = tournament_selection(self.population, fitness_scores, tournament_size=safe_tournament_size)
                selected_parents.append(parent)

            next_generation = create_next_generation(selected_parents, self.population_size, current_rate)

            # Elitism
            if elitism and self.best_sequence is not None:
                replace_idx = random.randint(0, self.population_size - 1)
                next_generation[replace_idx] = self.best_sequence.copy()

            self.population = next_generation

        # Apply Heavy Local Search once at the very end
        if use_heavy_ls:
            if callback:
                callback("HEAVY_SEARCH", n_iterations, self.best_score)
            
            final_improved_seq, final_improved_score = heavy_local_search(self.best_sequence, self.delayMatrix, self.total_processing_time)
            if final_improved_score < self.best_score:
                self.best_sequence = final_improved_seq
                self.best_score = final_improved_score
                self.convergence_history[-1] = self.best_score

        return self.best_sequence, self.best_score


class GA_PFSP_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GA for Permutation Flow Shop Scheduling")
        self.root.geometry("1400x950")

        self.time_matrix = None
        self.n_jobs = 0
        self.n_machines = 0
        self.ga_solver = None
        self.is_running = False

        self._setup_ui()
        self._load_default_data()

    def _setup_ui(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        control_frame = ttk.LabelFrame(main_container, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        self.info_text = tk.Text(control_frame, height=8, width=35, wrap=tk.WORD, font=('Courier', 9))
        self.info_text.grid(row=1, column=0, columnspan=2, pady=(0, 15))
        self.info_text.config(state=tk.DISABLED)

        ttk.Button(control_frame, text="📁 Load Data File", command=self._load_data_file).grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="GA Parameters", font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=2, pady=(10, 5))

        ttk.Label(control_frame, text="Population Size:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.pop_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.pop_var, width=15).grid(row=5, column=1, sticky=tk.E, pady=2)

        ttk.Label(control_frame, text="Generations:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.iter_var = tk.StringVar(value="500")
        ttk.Entry(control_frame, textvariable=self.iter_var, width=15).grid(row=6, column=1, sticky=tk.E, pady=2)

        ttk.Label(control_frame, text="Base Mutation Rate:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.base_mut_var = tk.StringVar(value="0.3")
        ttk.Entry(control_frame, textvariable=self.base_mut_var, width=15).grid(row=7, column=1, sticky=tk.E, pady=2)

        # Dynamic Mutation Toggle
        self.dynamic_mut_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Enable Dynamic Mutation", variable=self.dynamic_mut_var, command=self._toggle_dynamic_mut).grid(row=8, column=0, columnspan=2, pady=(6, 2), sticky=tk.W)

        # Max Mutation Rate (Hidden by Default)
        self.max_mut_label = ttk.Label(control_frame, text="Max Mutation Rate:")
        self.max_mut_label.grid(row=9, column=0, sticky=tk.W, pady=2)
        self.max_mut_var = tk.StringVar(value="0.95")
        self.max_mut_entry = ttk.Entry(control_frame, textvariable=self.max_mut_var, width=15)
        self.max_mut_entry.grid(row=9, column=1, sticky=tk.E, pady=2)
        
        self.max_mut_label.grid_remove()
        self.max_mut_entry.grid_remove()

        ttk.Label(control_frame, text="Extinction Interval (0=off):").grid(row=10, column=0, sticky=tk.W, pady=2)
        self.extinct_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.extinct_var, width=15).grid(row=10, column=1, sticky=tk.E, pady=2)

        # Local Search Options
        ttk.Label(control_frame, text="Local Search Parameters", font=('Arial', 10, 'bold')).grid(row=11, column=0, columnspan=2, pady=(15, 5))

        self.micro_search_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Enable Micro LS (Periodic)", variable=self.micro_search_var).grid(row=12, column=0, sticky=tk.W, pady=2)
        
        self.micro_interval_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.micro_interval_var, width=8).grid(row=12, column=1, sticky=tk.E, pady=2)
        ttk.Label(control_frame, text="interval").grid(row=12, column=1, sticky=tk.W, padx=(5,0))

        self.heavy_search_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Enable Heavy LS (At End)", variable=self.heavy_search_var).grid(row=13, column=0, columnspan=2, pady=(2, 10), sticky=tk.W)

        self.run_button = ttk.Button(control_frame, text="▶ Run GA Optimization", command=self._run_ga, style='Accent.TButton')
        self.run_button.grid(row=14, column=0, columnspan=2, pady=(15, 5), sticky=(tk.W, tk.E))

        self.stop_button = ttk.Button(control_frame, text="⏹ Stop", command=self._stop_ga, state=tk.DISABLED)
        self.stop_button.grid(row=15, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="Progress:").grid(row=16, column=0, columnspan=2, pady=(10, 2), sticky=tk.W)
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.grid(row=17, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(control_frame, text="Ready", font=('Arial', 9), foreground='green')
        self.status_label.grid(row=18, column=0, columnspan=2, pady=5)

        self.results_text = tk.Text(control_frame, height=10, width=35, wrap=tk.WORD, font=('Courier', 9))
        self.results_text.grid(row=19, column=0, columnspan=2)
        self.results_text.config(state=tk.DISABLED)

        viz_container = ttk.Frame(main_container)
        viz_container.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_container.columnconfigure(0, weight=1)

        gantt_frame = ttk.LabelFrame(viz_container, text="Gantt Chart - Production Schedule", padding="5")
        gantt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.gantt_figure = Figure(figsize=(10, 4), dpi=100)
        self.gantt_canvas = FigureCanvasTkAgg(self.gantt_figure, master=gantt_frame)
        self.gantt_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        convergence_frame = ttk.LabelFrame(viz_container, text="Convergence Plot - Avg Completion Time", padding="5")
        convergence_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.convergence_figure = Figure(figsize=(10, 4), dpi=100)
        self.convergence_canvas = FigureCanvasTkAgg(self.convergence_figure, master=convergence_frame)
        self.convergence_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self._draw_empty_gantt()
        self._draw_empty_convergence()

    def _toggle_dynamic_mut(self):
        if self.dynamic_mut_var.get():
            self.max_mut_label.grid()
            self.max_mut_entry.grid()
        else:
            self.max_mut_label.grid_remove()
            self.max_mut_entry.grid_remove()

    def _load_default_data(self):
        default_file = os.path.join(HERE, 'data', 'daten3ACustomer_200_10.txt')
        default_file = os.path.normpath(default_file)

        if os.path.exists(default_file):
            self._load_data(default_file)
        else:
            self.time_matrix = np.array([[2,1,2],[1,1,1],[3,2,3],[1,2,2]])
            self.n_jobs, self.n_machines = self.time_matrix.shape
            self._update_info_display()

    def _load_data_file(self):
        filename = filedialog.askopenfilename(
            title="Select PFSP Data File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=os.path.join(HERE, 'data')
        )

        if filename:
            self._load_data(filename)

    def _load_data(self, filepath):
        try:
            self.time_matrix, self.n_jobs, self.n_machines = load_problem_data(filepath)
            self._update_info_display()
            messagebox.showinfo("Success", f"Loaded problem:\n{self.n_jobs} jobs × {self.n_machines} machines")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")

    def _update_info_display(self):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)  # <--- Corrected

        info = f"Problem Type:\n  Permutation Flow Shop\n  Scheduling (PFSP)\n\n"
        info += f"Problem Size:\n  Jobs: {self.n_jobs}\n  Machines: {self.n_machines}\n\n"
        info += f"Objective:\n  Minimize Avg Completion\n  Time"

        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)

    def _draw_empty_gantt(self):
        self.gantt_figure.clear()
        ax = self.gantt_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Run optimization to see schedule', ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.gantt_canvas.draw()

    def _draw_empty_convergence(self):
        self.convergence_figure.clear()
        ax = self.convergence_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Run optimization to see convergence', ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.convergence_canvas.draw()

    def _draw_gantt_chart(self, schedule, sequence):
        self.gantt_figure.clear()
        ax = self.gantt_figure.add_subplot(111)
        colors = plt.cm.Set3(np.linspace(0, 1, max(1, self.n_jobs)))

        for i in range(len(schedule['jobs'])):
            job = schedule['jobs'][i]
            machine = schedule['machines'][i]
            start = schedule['start_times'][i]
            end = schedule['end_times'][i]
            duration = end - start

            ax.barh(machine, duration, left=start, height=0.6, color=colors[job % len(colors)], edgecolor='black')
            if self.n_jobs <= 50:
                ax.text(start + duration/2, machine, f'J{job+1}', ha='center', va='center', fontsize=7, fontweight='bold')

        ax.set_yticks(range(self.n_machines))
        ax.set_yticklabels([f'M{i+1}' for i in range(self.n_machines)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        
        display_seq = sequence if len(sequence) <= 20 else list(sequence[:20]) + ["..."]
        ax.set_title(f'Sequence: {display_seq}')
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        makespan = schedule.get('makespan', 0)
        ax.axvline(makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan}')
        ax.legend(loc='upper right')

        self.gantt_figure.tight_layout()
        self.gantt_canvas.draw()

    def _draw_convergence_plot(self, convergence_history):
        self.convergence_figure.clear()
        ax = self.convergence_figure.add_subplot(111)
        iterations = range(1, len(convergence_history) + 1)
        ax.plot(iterations, convergence_history, 'b-', linewidth=2, label='Avg Completion Time')
        ax.fill_between(iterations, convergence_history, alpha=0.3)

        if convergence_history:
            best_idx = int(np.argmin(convergence_history))
            best_value = convergence_history[best_idx]
            ax.plot(best_idx + 1, best_value, 'r*', markersize=12, label=f'Optimum: {best_value:.2f}')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Score (Lower is Better)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        self.convergence_figure.tight_layout()
        self.convergence_canvas.draw()

    def _run_ga(self):
        if self.is_running:
            return

        if self.time_matrix is None:
            messagebox.showwarning("No Data", "Please load problem data first!")
            return

        try:
            population_size = int(self.pop_var.get())
            n_iterations = int(self.iter_var.get())
            base_mut = float(self.base_mut_var.get())
            dynamic_mut = self.dynamic_mut_var.get()
            
            # Safeguard: if dynamic is disabled, just use base_mut to prevent crash if box is empty
            max_mut = float(self.max_mut_var.get()) if dynamic_mut else base_mut
            
            extinction_interval = int(self.extinct_var.get())
            use_micro = self.micro_search_var.get()
            micro_interval = int(self.micro_interval_var.get())
            use_heavy = self.heavy_search_var.get()
        except ValueError:
            messagebox.showerror("Invalid Parameters", "Please enter valid numeric parameters!")
            return

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Running...", foreground='orange')
        self.progress['value'] = 0
        self.progress['maximum'] = n_iterations

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

        thread = threading.Thread(
            target=self._ga_worker,
            args=(population_size, n_iterations, base_mut, max_mut, dynamic_mut, extinction_interval, use_micro, micro_interval, use_heavy)
        )
        thread.daemon = True
        thread.start()

    def _ga_worker(self, population_size, n_iterations, base_mut, max_mut, dynamic_mut, extinction_interval, use_micro, micro_interval, use_heavy):
        try:
            self.ga_solver = GA_PFSP_Solver(self.time_matrix, population_size=population_size, mutation_rate=base_mut)

            start_time = time.time()

            def cb(iteration, total, best_score):
                if not self.is_running:
                    return
                self.root.after(0, self._update_progress, iteration, total, best_score)

            best_seq, best_score = self.ga_solver.solve(
                n_iterations=n_iterations,
                tournament_size=4,
                use_micro_ls=use_micro,
                micro_interval=micro_interval,
                use_heavy_ls=use_heavy,
                callback=cb,
                extinction_interval=extinction_interval,
                dynamic_mutation=dynamic_mut,
                base_mutation_rate=base_mut,
                max_mutation_rate=max_mut,
                elitism=True
            )

            total_time = time.time() - start_time

            if self.is_running:
                self.root.after(0, self._finish_ga, total_time)

        except Exception as e:
            self.root.after(0, self._ga_error, str(e))

    def _update_progress(self, iteration, total, best_score):
        if iteration == "HEAVY_SEARCH":
            self.status_label.config(text=f"Applying Heavy Local Search... (Best Avg Time: {best_score:.2f})", foreground='blue')
            self.progress.config(mode='indeterminate')
            self.progress.start()
        else:
            self.progress.config(mode='determinate')
            self.progress.stop()
            self.progress['value'] = iteration
            self.status_label.config(text=f"Gen {iteration}/{total} | Best Avg Time: {best_score:.2f}")
            if iteration % 10 == 0 or iteration == total:
                self._draw_convergence_plot(self.ga_solver.convergence_history)

    def _finish_ga(self, total_time):
        self.is_running = False
        self.progress.stop()
        self.progress.config(mode='determinate')
        self.progress['value'] = self.progress['maximum']
        
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Completed!", foreground='green')

        best_sequence = self.ga_solver.best_sequence
        best_score = self.ga_solver.best_score
        schedule = self.ga_solver.get_schedule_details(best_sequence)

        self._draw_gantt_chart(schedule, best_sequence)
        self._draw_convergence_plot(self.ga_solver.convergence_history)

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        results = "="*35 + "\n"
        results += "OPTIMIZATION RESULTS\n"
        results += "="*35 + "\n\n"
        results += f"Execution Time: {total_time:.2f}s\n"
        results += f"Generations: {len(self.ga_solver.convergence_history)}\n\n"
        results += f"Avg Completion Time: {best_score:.2f}\n"
        results += f"Total Completion Time: {best_score * self.n_jobs:.2f}\n"
        results += f"Makespan: {schedule.get('makespan', 0)}\n\n"
        
        display_seq = best_sequence if len(best_sequence) <= 50 else list(best_sequence[:50]) + ["..."]
        results += "Optimal Sequence:\n"
        results += f"  {display_seq}\n\n"
        results += "="*35

        self.results_text.insert(1.0, results)
        self.results_text.config(state=tk.DISABLED)

        messagebox.showinfo("Success", f"Optimization completed!\n\nBest Avg Time: {best_score:.2f}\nTime: {total_time:.2f}s")

    def _stop_ga(self):
        self.is_running = False
        self.progress.stop()
        self.progress.config(mode='determinate')
        self.status_label.config(text="Stopped", foreground='red')
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def _ga_error(self, error_msg):
        self.is_running = False
        self.progress.stop()
        self.progress.config(mode='determinate')
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Error!", foreground='red')
        messagebox.showerror("Error", f"GA optimization failed:\n{error_msg}")


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass

    app = GA_PFSP_GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()