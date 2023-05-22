import torch
import random

from typing import List
from torch.nn import Module
from enviroment import SnakeEnviroment
from misc import Direction
from model import SnakeNeural
from selection import roulette_wheel_selection, elitism_selection
from crossover import single_point_crossover
from mutate import mutate


class GenericAlgoOptimizer:
    def __init__(s, env: SnakeEnviroment, model: Module):
        s.env = env
        s.model = model

        pass

    def optimize(s, generations=100) -> Module:
        s.population = s.generate_population(s.population_size)
        s.fitness = [s.calculate_fitness(individual) for individual in s.population]

        while generations >= 0:
            parents = s.selection(s.population, s.fitness, s.selected_size)
            s.population = s.breed(parents)

            for j in range(len(s.population)):
                if random.random() < s.mutation_rate:
                    s.population[j] = s.mutation(s.population[j])

            s.fitness = [s.calculate_fitness(individual) for individual in s.population]
            print(f"Generation {generations}: Best fitness {max(s.fitness)}")

            # generations -= 1

        return elitism_selection(s.population, s.fitness, 1)[0]

    def generate_population(s, population_size: int) -> List[Module]:
        population = [type(s.model)() for _ in range(population_size)]
        return population

    def calculate_fitness(s, individual: SnakeNeural) -> float:
        s.env.initialize()

        state = s.env.get_state()
        game_over = False

        while not game_over:
            action = individual(state)
            action = torch.argmax(action).item()

            s.env.step(list(Direction)[action])

            frame, reward, score, game_over = s.env.get_info()
            next_state = s.env.get_state()

            # individual.traine(state, action, reward, next_state, game_over)

            state = next_state

        # fitness = s.fitness_func(s.env.frames, score)

        return s.fitness_func(frame, score)

    def selection(
        s,
        population: List[Module],
        fitness: List[float],
        selected_size: int,
    ) -> List[Module]:
        best_childs, best_childs_indices = roulette_wheel_selection(
            population,
            fitness,
            selected_size,
        )

        return best_childs

    def breed(s, parents: List[Module]) -> List[Module]:
        new_population = []

        while len(new_population) < s.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = s.crossover(parent1, parent2)

            new_population.append(child1)

            if len(new_population) < s.population_size:
                new_population.append(child2)

        return new_population

    def crossover(s, parent1: Module, parent2: Module) -> List[Module]:
        return single_point_crossover(parent1, parent2)

    def mutation(s, individual: Module) -> Module:
        model = mutate(individual, s.mutation_rate, s.mutation_std)
        return model

    def fitness_func(s, frames: int, score: int) -> float:
        fitness = (
            frames
            + ((2**score) + (score**2.1) * 500)
            - (((0.25 * frames) ** 1.3) * (score**1.2))
        )
        # fitness = (frames) + ((2**score) + (score**2.1)*500) - (((.25 * frames)) * (score))
        return max(fitness, 0.1)
