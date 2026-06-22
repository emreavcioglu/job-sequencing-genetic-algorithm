import os
import random
import numpy as np


def load_customers(data_dir):
    customers_path = os.path.join(data_dir, "daten3ACustomer_200_10.txt")
    with open(customers_path, "r") as f:
        jobs = int(f.readline().strip())
        machines = int(f.readline().strip())
        customers = [list(map(int, f.readline().strip().split())) for _ in range(jobs)]

    timeMatrix = np.array(customers, dtype=int)
    return jobs, machines, customers, timeMatrix


def precompute_delays(timeMatrix):
    n, m = timeMatrix.shape
    delayMatrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            max_delay = 0
            for j in range(1, m + 1):
                sum_i = np.sum(timeMatrix[i, :j])
                sum_k = np.sum(timeMatrix[k, :j-1])
                delay = sum_i - sum_k
                if delay > max_delay:
                    max_delay = delay

            delayMatrix[i, k] = max_delay

    total_processing_time = np.sum(timeMatrix)
    return delayMatrix, total_processing_time


def initialize_population(jobs, population_size):
    job_indices = list(range(jobs))
    return [np.random.permutation(job_indices) for _ in range(population_size)]


def calculate_average_completion_time(sequence, delayMatrix, total_processing_time):
    n = len(sequence)
    total_delay_term = 0

    for i in range(n - 1):
        job_i = sequence[i]
        job_k = sequence[i + 1]
        weight = n - 1 - i
        total_delay_term += weight * delayMatrix[job_i, job_k]

    return (total_delay_term + total_processing_time) / n


def evaluate_fitness(population, delayMatrix, total_processing_time):
    return [
        -calculate_average_completion_time(individual, delayMatrix, total_processing_time)
        for individual in population
    ]


def two_way_tournament(fitness_list, population_working):
    idx1, idx2 = random.sample(range(len(fitness_list)), 2)
    loser_idx = idx1 if fitness_list[idx1] < fitness_list[idx2] else idx2

    fitness_list.pop(loser_idx)
    population_working.pop(loser_idx)
    return fitness_list, population_working


def three_way_tournament(fitness_list):
    selected = random.sample(fitness_list, 3)
    killed_fitness = fitness_list.pop(fitness_list.index(min(selected)))
    return killed_fitness


def proportional_selection(fitness_list, population_working):
    total_fitness = sum(fitness_list)
    if total_fitness == 0:
        selected_index = random.choice(range(len(fitness_list)))
        fitness_list.pop(selected_index)
        population_working.pop(selected_index)
        return fitness_list, population_working

    best_fitness = max(fitness_list)
    p_death_list = [
        (fitness_list[i] - best_fitness) ** 2 / (total_fitness - fitness_list[i] + 1) ** 2
        for i in range(len(fitness_list))
    ]

    selected_index = random.choices(range(len(fitness_list)), weights=p_death_list, k=1)[0]
    fitness_list.pop(selected_index)
    population_working.pop(selected_index)
    return fitness_list, population_working


def tournament_selection(population, fitness_scores, tournament_size=5):
    contestant_indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(contestant_indices, key=lambda idx: fitness_scores[idx])
    return population[best_idx]


def crossover(parent1, parent2):
    n = len(parent1)
    cut1, cut2 = sorted(random.sample(range(n), 2))

    def ox(p1, p2):
        child = np.full(n, -1, dtype=int)
        child[cut1:cut2 + 1] = p1[cut1:cut2 + 1]
        segment_values = set(p1[cut1:cut2 + 1])
        p2_remaining = [gene for gene in p2 if gene not in segment_values]

        pos = [
            i % n
            for i in range(cut2 + 1, cut2 + 1 + n)
            if (i % n) < cut1 or (i % n) > cut2
        ]

        for idx, gene in zip(pos, p2_remaining):
            child[idx] = gene

        return np.array(child)

    return ox(parent1, parent2), ox(parent2, parent1)


def inversion_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        n = len(individual)
        cut1, cut2 = sorted(random.sample(range(n), 2))
        individual = individual.copy()
        individual[cut1:cut2 + 1] = individual[cut1:cut2 + 1][::-1]

    return individual


def insertion_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        individual = individual.tolist()
        idx_from, idx_to = random.sample(range(len(individual)), 2)
        job = individual.pop(idx_from)
        individual.insert(idx_to, job)
        return np.array(individual)

    return individual


def get_best_solution(population, fitness_list, delayMatrix, total_processing_time):
    best_idx = np.argmax(fitness_list)
    best_individual = population[best_idx]
    best_makespan = calculate_average_completion_time(
        best_individual, delayMatrix, total_processing_time
    )
    return best_individual, best_makespan


def micro_local_search(sequence, jobs_to_check, delayMatrix, total_processing_time):
    best_seq = sequence.tolist()
    best_score = calculate_average_completion_time(
        best_seq, delayMatrix, total_processing_time
    )
    indices_to_test = random.sample(range(len(best_seq)), jobs_to_check)

    for i in indices_to_test:
        job_to_move = best_seq.pop(i)
        best_insert_pos = i

        for j in range(len(best_seq) + 1):
            best_seq.insert(j, job_to_move)
            new_score = calculate_average_completion_time(
                best_seq, delayMatrix, total_processing_time
            )

            if new_score < best_score:
                best_score = new_score
                best_insert_pos = j

            best_seq.pop(j)

        best_seq.insert(best_insert_pos, job_to_move)

    return np.array(best_seq), best_score


def heavy_local_search(sequence, delayMatrix, total_processing_time):
    best_seq = sequence.tolist()
    best_score = calculate_average_completion_time(
        best_seq, delayMatrix, total_processing_time
    )
    improved = True

    while improved:
        improved = False
        for i in range(len(best_seq)):
            job_to_move = best_seq.pop(i)

            for j in range(len(best_seq) + 1):
                best_seq.insert(j, job_to_move)
                new_score = calculate_average_completion_time(
                    best_seq, delayMatrix, total_processing_time
                )

                if new_score < best_score:
                    best_score = new_score
                    improved = True
                    break
                else:
                    best_seq.pop(j)

            if improved:
                break
            else:
                best_seq.insert(i, job_to_move)

    return np.array(best_seq), best_score


def create_next_generation(selected_parents, population_size, mutation_rate):
    next_generation = []
    parents = selected_parents.copy()
    random.shuffle(parents)

    while len(next_generation) < population_size:
        p1 = parents[len(next_generation) % len(parents)]
        p2 = parents[(len(next_generation) + 1) % len(parents)]

        child1, child2 = crossover(p1, p2)
        child1 = insertion_mutation(child1, mutation_rate)
        child2 = insertion_mutation(child2, mutation_rate)

        next_generation.append(child1)
        if len(next_generation) < population_size:
            next_generation.append(child2)

    return next_generation


def dynamic_optimizer(stagnation_counter, current_rate, base_rate, max_rate):
    if stagnation_counter == 0:
        return base_rate
    if stagnation_counter > 30:
        return min(current_rate + 0.05, max_rate)
    return current_rate
