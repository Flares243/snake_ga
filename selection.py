import random
from copy import deepcopy
from typing import List

import numpy as np
from torch.nn import Module


def roulette_wheel_selection(
    population: List[Module],
    fitness: List[float],
    num_childs: int,
) -> tuple[List[Module], List[int]]:
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]

    wheel = np.cumsum(probabilities)
    selected = []
    selected_indices = []

    for _ in range(num_childs):
        spin = random.random()
        for j in range(len(wheel)):
            if spin <= wheel[j]:
                selected.append(population[j])
                selected_indices.append(j)
                break

    return (selected, selected_indices)


def elitism_selection(
    population: List[Module],
    fitness: List[float],
    select_size: int,
) -> tuple[List[Module], List[int]]:
    best_population = [0] * select_size
    indices = [0] * select_size

    fitness_copy = deepcopy(fitness)

    for index in range(select_size):
        best_fitness_index = fitness_copy.index(max(fitness_copy))
        fitness_copy[best_fitness_index] = -999

        indices[index] = best_fitness_index
        best_population[index] = population[best_fitness_index]

    return (best_population, indices)
