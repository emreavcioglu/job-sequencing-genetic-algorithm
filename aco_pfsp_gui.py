"""
GUI Application for Ant Colony Optimization - PFSP
Comprehensive visualization with Gantt charts, convergence plots, and real-time progress
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time

# Import ACO solver
from aco_pfsp_solver import ACO_PFSP, load_problem_data


class ACO_PFSP_GUI:
    """
    Comprehensive GUI for ACO-based PFSP solver
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("ACO for Permutation Flow Shop Scheduling - JIS Production Planning")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Data
        self.time_matrix = None
        self.n_jobs = 0
        self.n_machines = 0
        self.aco_solver = None
        self.is_running = False
        
        # Setup UI
        self._setup_ui()
        
        # Load default data
        self._load_default_data()
    
    def _setup_ui(self):
        """Setup the user interface"""
        
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # === LEFT PANEL: Controls ===
        control_frame = ttk.LabelFrame(main_container, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Problem info
        ttk.Label(control_frame, text="Problem Information", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        self.info_text = tk.Text(control_frame, height=8, width=35, wrap=tk.WORD, font=('Courier', 9))
        self.info_text.grid(row=1, column=0, columnspan=2, pady=(0, 15))
        self.info_text.config(state=tk.DISABLED)
        
        # Load data button
        ttk.Button(control_frame, text="📁 Load Data File", command=self._load_data_file).grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=2, pady=15, sticky=(tk.W, tk.E))
        
        # ACO Parameters
        ttk.Label(control_frame, text="ACO Parameters", font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        # Number of ants
        ttk.Label(control_frame, text="Number of Ants:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.ants_var = tk.StringVar(value="20")
        ttk.Entry(control_frame, textvariable=self.ants_var, width=15).grid(row=5, column=1, sticky=tk.E, pady=2)
        
        # Iterations
        ttk.Label(control_frame, text="Iterations:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.iterations_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.iterations_var, width=15).grid(row=6, column=1, sticky=tk.E, pady=2)
        
        # Alpha
        ttk.Label(control_frame, text="Alpha (pheromone):").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.alpha_var = tk.StringVar(value="1.0")
        ttk.Entry(control_frame, textvariable=self.alpha_var, width=15).grid(row=7, column=1, sticky=tk.E, pady=2)
        
        # Beta
        ttk.Label(control_frame, text="Beta (heuristic):").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.beta_var = tk.StringVar(value="2.0")
        ttk.Entry(control_frame, textvariable=self.beta_var, width=15).grid(row=8, column=1, sticky=tk.E, pady=2)
        
        # Evaporation rate
        ttk.Label(control_frame, text="Evaporation (ρ):").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.rho_var = tk.StringVar(value="0.1")
        ttk.Entry(control_frame, textvariable=self.rho_var, width=15).grid(row=9, column=1, sticky=tk.E, pady=2)
        
        # Local search
        self.local_search_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Enable Local Search", variable=self.local_search_var).grid(row=10, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=2, pady=15, sticky=(tk.W, tk.E))
        
        # Run button
        self.run_button = ttk.Button(control_frame, text="▶ Run ACO Optimization", command=self._run_aco, style='Accent.TButton')
        self.run_button.grid(row=12, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Stop button
        self.stop_button = ttk.Button(control_frame, text="⏹ Stop", command=self._stop_aco, state=tk.DISABLED)
        self.stop_button.grid(row=13, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Progress
        ttk.Label(control_frame, text="Progress:", font=('Arial', 9)).grid(row=14, column=0, columnspan=2, pady=(15, 5), sticky=tk.W)
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.grid(row=15, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(control_frame, text="Ready", font=('Arial', 9), foreground='green')
        self.status_label.grid(row=16, column=0, columnspan=2, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=17, column=0, columnspan=2, pady=15, sticky=(tk.W, tk.E))
        
        # Results
        ttk.Label(control_frame, text="Results", font=('Arial', 10, 'bold')).grid(row=18, column=0, columnspan=2, pady=(0, 10))
        
        self.results_text = tk.Text(control_frame, height=10, width=35, wrap=tk.WORD, font=('Courier', 9))
        self.results_text.grid(row=19, column=0, columnspan=2)
        self.results_text.config(state=tk.DISABLED)
        
        # === RIGHT PANEL: Visualizations ===
        viz_container = ttk.Frame(main_container)
        viz_container.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_container.columnconfigure(0, weight=1)
        viz_container.rowconfigure(0, weight=1)
        viz_container.rowconfigure(1, weight=1)
        
        # Top: Gantt Chart
        gantt_frame = ttk.LabelFrame(viz_container, text="Gantt Chart - Production Schedule", padding="5")
        gantt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        gantt_frame.columnconfigure(0, weight=1)
        gantt_frame.rowconfigure(0, weight=1)
        
        self.gantt_figure = Figure(figsize=(10, 4), dpi=100)
        self.gantt_canvas = FigureCanvasTkAgg(self.gantt_figure, master=gantt_frame)
        self.gantt_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bottom: Convergence Plot
        convergence_frame = ttk.LabelFrame(viz_container, text="Convergence Plot - Algorithm Progress", padding="5")
        convergence_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        convergence_frame.columnconfigure(0, weight=1)
        convergence_frame.rowconfigure(0, weight=1)
        
        self.convergence_figure = Figure(figsize=(10, 4), dpi=100)
        self.convergence_canvas = FigureCanvasTkAgg(self.convergence_figure, master=convergence_frame)
        self.convergence_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial empty plots
        self._draw_empty_gantt()
        self._draw_empty_convergence()
    
    def _load_default_data(self):
        """Load default problem data"""
        default_file = "job-sequencing-genetic-algorithm-main/data/datenCustomer_5_3.txt"
        
        if os.path.exists(default_file):
            self._load_data(default_file)
        else:
            # Use hardcoded small problem
            self.time_matrix = np.array([
                [2, 1, 2],
                [1, 1, 1],
                [3, 2, 3],
                [1, 2, 2]
            ])
            self.n_jobs, self.n_machines = self.time_matrix.shape
            self._update_info_display()
    
    def _load_data_file(self):
        """Load problem data from file"""
        filename = filedialog.askopenfilename(
            title="Select PFSP Data File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir="job-sequencing-genetic-algorithm-main/data"
        )
        
        if filename:
            self._load_data(filename)
    
    def _load_data(self, filepath):
        """Load data from file"""
        try:
            self.time_matrix, self.n_jobs, self.n_machines = load_problem_data(filepath)
            self._update_info_display()
            messagebox.showinfo("Success", f"Loaded problem:\n{self.n_jobs} jobs × {self.n_machines} machines")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
    
    def _update_info_display(self):
        """Update problem information display"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        info = f"Problem Type:\n  Permutation Flow Shop\n  Scheduling (PFSP)\n\n"
        info += f"Problem Size:\n  Jobs: {self.n_jobs}\n  Machines: {self.n_machines}\n\n"
        info += f"Objective:\n  Minimize Total\n  Completion Time"
        
        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)
    
    def _draw_empty_gantt(self):
        """Draw empty Gantt chart placeholder"""
        self.gantt_figure.clear()
        ax = self.gantt_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Run optimization to see schedule', 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.gantt_canvas.draw()
    
    def _draw_empty_convergence(self):
        """Draw empty convergence plot placeholder"""
        self.convergence_figure.clear()
        ax = self.convergence_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Run optimization to see convergence', 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.convergence_canvas.draw()
    
    def _draw_gantt_chart(self, schedule, sequence):
        """Draw Gantt chart of the schedule"""
        self.gantt_figure.clear()
        ax = self.gantt_figure.add_subplot(111)
        
        # Color palette for jobs
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_jobs))
        
        # Plot schedule
        for i in range(len(schedule['jobs'])):
            job = schedule['jobs'][i]
            machine = schedule['machines'][i]
            start = schedule['start_times'][i]
            end = schedule['end_times'][i]
            duration = end - start
            
            ax.barh(machine, duration, left=start, height=0.6, 
                   color=colors[job], edgecolor='black', linewidth=1)
            
            # Add job label
            ax.text(start + duration/2, machine, f'J{job+1}', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_yticks(range(self.n_machines))
        ax.set_yticklabels([f'M{i+1}' for i in range(self.n_machines)])
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('Machines', fontsize=10, fontweight='bold')
        ax.set_title(f'Production Schedule | Sequence: {[j+1 for j in sequence]}', 
                    fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_ylim(-0.5, self.n_machines - 0.5)
        
        # Add makespan line
        makespan = max(schedule['end_times'])
        ax.axvline(makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan}')
        ax.legend(loc='upper right')
        
        self.gantt_figure.tight_layout()
        self.gantt_canvas.draw()
    
    def _draw_convergence_plot(self, convergence_history):
        """Draw convergence plot"""
        self.convergence_figure.clear()
        ax = self.convergence_figure.add_subplot(111)
        
        iterations = range(1, len(convergence_history) + 1)
        
        ax.plot(iterations, convergence_history, 'b-', linewidth=2, label='Best Makespan')
        ax.fill_between(iterations, convergence_history, alpha=0.3)
        
        # Mark best solution
        best_idx = np.argmin(convergence_history)
        best_value = convergence_history[best_idx]
        ax.plot(best_idx + 1, best_value, 'r*', markersize=15, 
               label=f'Optimum: {best_value:.2f} (Iter {best_idx + 1})')
        
        ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax.set_ylabel('Makespan (Avg Completion Time)', fontsize=10, fontweight='bold')
        ax.set_title('ACO Convergence - Solution Quality over Time', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        
        self.convergence_figure.tight_layout()
        self.convergence_canvas.draw()
    
    def _run_aco(self):
        """Run ACO optimization in separate thread"""
        if self.is_running:
            return
        
        if self.time_matrix is None:
            messagebox.showwarning("No Data", "Please load problem data first!")
            return
        
        # Get parameters
        try:
            n_ants = int(self.ants_var.get())
            n_iterations = int(self.iterations_var.get())
            alpha = float(self.alpha_var.get())
            beta = float(self.beta_var.get())
            rho = float(self.rho_var.get())
            use_local_search = self.local_search_var.get()
        except ValueError:
            messagebox.showerror("Invalid Parameters", "Please enter valid numeric parameters!")
            return
        
        # Update UI
        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Running...", foreground='orange')
        self.progress['value'] = 0
        self.progress['maximum'] = n_iterations
        
        # Clear results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Run in thread
        thread = threading.Thread(target=self._aco_worker, 
                                  args=(n_ants, n_iterations, alpha, beta, rho, use_local_search))
        thread.daemon = True
        thread.start()
    
    def _aco_worker(self, n_ants, n_iterations, alpha, beta, rho, use_local_search):
        """ACO worker thread"""
        try:
            # Initialize ACO
            self.aco_solver = ACO_PFSP(
                time_matrix=self.time_matrix,
                n_ants=n_ants,
                n_iterations=n_iterations,
                alpha=alpha,
                beta=beta,
                rho=rho,
                q=100.0,
                elite_weight=2.0
            )
            
            # Custom solve with progress updates
            heuristic = self.aco_solver._calculate_heuristic()
            start_time = time.time()
            
            for iteration in range(n_iterations):
                if not self.is_running:
                    break
                
                # Each ant constructs solution
                all_sequences = []
                all_makespans = []
                
                for ant in range(n_ants):
                    sequence = self.aco_solver._construct_solution(heuristic)
                    
                    if use_local_search and np.random.random() < 0.3:
                        sequence = self.aco_solver._local_search_2opt(sequence)
                    
                    makespan = self.aco_solver.calculate_makespan(sequence)
                    all_sequences.append(sequence)
                    all_makespans.append(makespan)
                    
                    if makespan < self.aco_solver.best_makespan:
                        self.aco_solver.best_makespan = makespan
                        self.aco_solver.best_sequence = sequence.copy()
                
                self.aco_solver._update_pheromones(all_sequences, all_makespans)
                
                iter_best = min(all_makespans)
                self.aco_solver.iteration_best_makespans.append(iter_best)
                self.aco_solver.convergence_history.append(self.aco_solver.best_makespan)
                
                # Update UI
                self.root.after(0, self._update_progress, iteration + 1, n_iterations, self.aco_solver.best_makespan)
            
            total_time = time.time() - start_time
            
            # Final update
            if self.is_running:
                self.root.after(0, self._finish_aco, total_time)
            
        except Exception as e:
            self.root.after(0, self._aco_error, str(e))
    
    def _update_progress(self, iteration, total, best_makespan):
        """Update progress bar and status"""
        self.progress['value'] = iteration
        self.status_label.config(text=f"Iteration {iteration}/{total} | Best: {best_makespan:.2f}")
        
        # Update convergence plot in real-time
        if iteration % 10 == 0 or iteration == total:
            self._draw_convergence_plot(self.aco_solver.convergence_history)
    
    def _finish_aco(self, total_time):
        """Finish ACO and display results"""
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Completed!", foreground='green')
        
        # Get results
        best_sequence = self.aco_solver.best_sequence
        best_makespan = self.aco_solver.best_makespan
        schedule = self.aco_solver.get_schedule_details(best_sequence)
        
        # Draw visualizations
        self._draw_gantt_chart(schedule, best_sequence)
        self._draw_convergence_plot(self.aco_solver.convergence_history)
        
        # Display results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        results = "="*35 + "\n"
        results += "OPTIMIZATION RESULTS\n"
        results += "="*35 + "\n\n"
        results += f"Execution Time: {total_time:.2f}s\n"
        results += f"Iterations: {len(self.aco_solver.convergence_history)}\n\n"
        results += f"Best Makespan: {best_makespan:.2f}\n"
        results += f"Total Makespan: {max(schedule['end_times'])}\n\n"
        results += "Optimal Sequence:\n"
        results += f"  {[j+1 for j in best_sequence]}\n\n"
        results += "Job Completion Times:\n"
        for job_idx in best_sequence:
            ct = schedule['job_completion_times'][job_idx]
            results += f"  Job {job_idx+1}: {ct}\n"
        results += "\n" + "="*35
        
        self.results_text.insert(1.0, results)
        self.results_text.config(state=tk.DISABLED)
        
        messagebox.showinfo("Success", f"Optimization completed!\n\nBest Makespan: {best_makespan:.2f}\nTime: {total_time:.2f}s")
    
    def _stop_aco(self):
        """Stop ACO optimization"""
        self.is_running = False
        self.status_label.config(text="Stopped", foreground='red')
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def _aco_error(self, error_msg):
        """Handle ACO error"""
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Error!", foreground='red')
        messagebox.showerror("Error", f"ACO optimization failed:\n{error_msg}")


def main():
    """Main entry point for GUI application"""
    root = tk.Tk()
    
    # Configure ttk style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create application
    app = ACO_PFSP_GUI(root)
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
