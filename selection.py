from typing import List
from torch.nn import Module

import numpy as np
import random


def roulette_wheel_selection(
    population: List[Module], fitness: List[float], num_childs: int
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
    population: List[Module], fitness: List[float], num_individuals: int
) -> List[Module]:
    sorted_fitness_index = sorted(
        range(len(fitness)), key=lambda x: fitness[x], reverse=True
    )

    sorted_population = [population[i] for i in sorted_fitness_index]

    return sorted_population[:num_individuals]
