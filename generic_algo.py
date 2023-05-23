from random import sample
from typing import List

from torch.nn import Module

from crossover import random_crossover
from enviroment import SnakeEnviroment
from misc import PARENTS_SIZE, POPULATION_SIZE, Direction
from model import SnakeNeural
from mutate import mutation
from selection import elitism_selection, roulette_wheel_selection


class GenericAlgoOptimizer:
    def __init__(
        s,
        env: SnakeEnviroment,
        model: Module,
        init_population: List[Module] = [],
    ):
        s.env = env
        s.model = model
        s.population = init_population

    def optimize(s, generations=100) -> Module:
        s.generate_population()
        s.fitness = [s.calculate_fitness(individual) for individual in s.population]

        while generations > 0:
            parents, parents_fitness = s.selection(s.population, s.fitness)
            new_population = s.breed(parents)

            new_population_fitness = [
                s.calculate_fitness(individual) for individual in new_population
            ]

            print(
                f"Generation {generations} | Best fitness {max(s.fitness)} | Avg best fitness {sum(parents_fitness) / len(parents_fitness)}"
            )

            s.population = parents + new_population
            s.fitness = parents_fitness + new_population_fitness

            # generations -= 1

        return elitism_selection(s.population, s.fitness)

    def generate_population(s) -> List[Module]:
        while len(s.population) < POPULATION_SIZE:
            s.population.append(type(s.model)())

    def calculate_fitness(s, individual: SnakeNeural) -> float:
        s.env.initialize()

        state = s.env.get_state()
        game_over = False

        while not game_over:
            pred = individual(state)
            pred = pred.data.tolist()

            action = pred.index(max(pred))

            s.env.step(list(Direction)[action])

            frames, reward, score, game_over = s.env.get_info()
            next_state = s.env.get_state()

            state = next_state

        return s.fitness_func(frames, score)

    def selection(
        s,
        population: List[Module],
        fitness: List[float],
    ) -> tuple[List[Module], List[float]]:
        best_population, indices = roulette_wheel_selection(
            population,
            fitness,
            PARENTS_SIZE,
        )

        best_fitness = [fitness[i] for i in indices]

        return best_population, best_fitness

    def crossover(s, parent1: Module, parent2: Module) -> List[Module]:
        return random_crossover(parent1, parent2)

    def mutation(s, individual: Module) -> Module:
        return mutation(individual)

    def breed(s, parents: List[Module]) -> List[Module]:
        population = []

        while len(population) < POPULATION_SIZE - PARENTS_SIZE:
            parent1, parent2 = sample(parents, k=2)
            child = s.crossover(parent1, parent2)
            child = s.mutation(child)
            population.append(child)

            if len(population) < POPULATION_SIZE - PARENTS_SIZE:
                child = s.crossover(parent2, parent1)
                child = s.mutation(child)
                population.append(child)

        return population

    def fitness_func(s, frames: int, score: int) -> float:
        fitness = (
            frames
            + ((2**score) + (score**2.1) * 500)
            - (((0.25 * frames) ** 1.3) * (score**1.2))
        )
        # fitness = (frames) + ((2**score) + (score**2.1)*500) - (((.25 * frames)) * (score))
        return max(fitness, 0.1)
