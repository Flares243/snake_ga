import torch
import random

from typing import List
from torch.nn import Module
from enviroment import SnakeEnviroment
from selection import roulette_wheel_selection, elitism_selection
from crossover import single_point_crossover
from mutate import mutate


class GenericAlgoOptimizer:
    def __init__(s, env: SnakeEnviroment, model: Module):
        s.env = env
        s.model = model
        s.population_size = 100
        s.mutation_rate = 0.1
        s.mutation_std = 0.1
        s.selected_size = 10
        pass

    def optimize(s, generations=500) -> Module:
        s.population = s.generate_population(s.population_size)
        s.fitness = [s.calculate_fitness(individual) for individual in s.population]

        for i in range(generations):
            parents = s.selection(s.population, s.fitness, s.selected_size)
            s.population = s.breed(parents)

            for i in range(len(s.population)):
                if random.random() < s.mutation_rate:
                    s.population[i] = s.mutation(s.population[i])

            s.fitness = [s.calculate_fitness(individual) for individual in s.population]
            print(f"Generation {i+1}: Best fitness {max(s.fitness)}")

        return elitism_selection(s.population, s.fitness, 1)[0]

    def generate_population(s, population_size) -> List[Module]:
        population = []

        for _ in range(population_size):
            model = type(s.model)()
            for param in model.parameters():
                param.data = torch.randn_like(param.data)
            population.append(model)

        return population

    def calculate_fitness(s, individual: Module) -> float:
        s.env.initialize()

        state = s.env.get_obs()
        total_reward = 0
        game_over = False

        while not game_over:
            action = individual(state)
            action = torch.argmax(action).item()
            s.env.step(action)
            reward, score, game_over = s.env.get_info()
            # s.env.render()
            state = s.env.get_obs()
            total_reward += reward

        return total_reward

    def selection(
        s,
        population: List[Module],
        fitness: List[float],
        selected_size: int,
    ) -> List[Module]:
        best_childs = roulette_wheel_selection(
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
