from typing import List
from torch.nn import Module

import numpy as np
import random


def roulette_wheel_selection(
    population: List[Module], fitness: List[float], num_childs: int
) -> List[Module]:
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]

    wheel = np.cumsum(probabilities)
    selected = []

    for _ in range(num_childs):
        spin = random.random()
        for j in range(len(wheel)):
            if spin <= wheel[j]:
                selected.append(population[j])
                break

    return selected


def elitism_selection(
    population: List[Module], fitness: List[float], num_individuals: int
) -> List[Module]:
    sorted_population = [p for _, p in sorted(zip(fitness, population), reverse=True)]

    return sorted_population[:num_individuals]
