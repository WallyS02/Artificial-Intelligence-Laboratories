from itertools import compress, combinations
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def crossover(parent1, parent2):
    child = []
    for j in range(int(len(parent1)/2)):
        child.append(parent1[j])
    for j in range(int(len(parent2) / 2), len(parent2)):
        child.append(parent1[j])
    for j in range(len(child)):
        if random.random() < mutation_rate:
            child[j] = not child[j]
    return child


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1
mutation_rate = 0.01
mutation_number = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    probability = []
    fitness_sum = 0
    for individual in population:
        fitness_sum = fitness_sum + fitness(items, knapsack_max_capacity, individual)
    for individual in population:
        probability.append(fitness(items, knapsack_max_capacity, individual) / fitness_sum)
    chosen_ones = random.choices(
        population,
        weights=probability,
        k=n_selection,
    )
    fitness_array = [fitness(items, knapsack_max_capacity, ind) for ind in population]
    elite = []
    for _ in range(n_elite):
        idx = fitness_array.index(max(fitness_array))
        elite.append(population[idx])
        fitness_array.pop(idx)
        population.pop(idx)
    pairs = list(combinations(chosen_ones, 2))
    children = []
    for parents in pairs:
        parent1 = parents[0]
        parent2 = parents[1]
        children.append(crossover(parent1, parent2))
        children.append(crossover(parent2, parent1))
    population = []
    for i in range(population_size - n_elite):
        population.append(children[i])
    for elites in elite:
        population.append(elites)

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
