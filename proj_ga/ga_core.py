import random
from typing import Callable, List, Tuple

Chromosome = List[int]
Population = List[Chromosome]


def create_individual(length: int) -> Chromosome:
    return [random.randint(0, 1) for _ in range(length)]


def create_population(pop_size: int, chromosome_length: int) -> Population:
    return [create_individual(chromosome_length) for _ in range(pop_size)]


def roulette_selection(
    population: Population,
    fitness_function: Callable[[Chromosome], float]
) -> Chromosome:
    fitness_values = [fitness_function(ind) for ind in population]
    total_fitness = sum(fitness_values)

    if total_fitness == 0:
        return random.choice(population).copy()

    pick = random.uniform(0, total_fitness)
    current = 0

    for individual, fit in zip(population, fitness_values):
        current += fit
        if current >= pick:
            return individual.copy()

    return population[-1].copy()


def one_point_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    crossover_rate: float
) -> Tuple[Chromosome, Chromosome]:
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def bit_flip_mutation(
    individual: Chromosome,
    mutation_rate: float
) -> Chromosome:
    mutated = individual.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]

    return mutated


def evolve_population(
    population: Population,
    fitness_function: Callable[[Chromosome], float],
    crossover_rate: float,
    mutation_rate: float,
    pop_size: int
) -> Population:
    new_population = []

    while len(new_population) < pop_size:
        parent1 = roulette_selection(population, fitness_function)
        parent2 = roulette_selection(population, fitness_function)

        child1, child2 = one_point_crossover(parent1, parent2, crossover_rate)

        child1 = bit_flip_mutation(child1, mutation_rate)
        child2 = bit_flip_mutation(child2, mutation_rate)

        new_population.append(child1)

        if len(new_population) < pop_size:
            new_population.append(child2)

    return new_population