import numpy as np
import random

from model import SnakeNeural
from enviroment import SnakeEnviroment


class GenericAlgoOptimizer:
    def __init__(s, env: SnakeEnviroment, model: SnakeNeural):
        s.env = env
        s.model = model
        s.generations = 500
        s.population_size = 500
        pass

    def optimize(s):
        # Intitialize the population with random weights
        population = s.generate_population()

        # For each generation...
        for _ in range(s.generations):
            # Calculate fitness of each individual
            fitness = [s.calculate_fitness(individual) for individual in population]
            # Select some best individuals for breeding
            parents = s.select_best(population, fitness, s.population_size // 2)
            # Breeding... (⁄ ⁄>⁄ ω ⁄<⁄ ⁄)
            population = s.breed(parents)

        # The best individual will be in population[0]
        s.model.model.set_weights(population[0])
        s.model.model.save_weights("snake_ai.pth")

    def generate_population(s):
        population = []

        for _ in range(s.population_size):
            individual = []

            for index in range(len(s.model.model.get_weights())):
                shape = s.model.model.get_weights()[index].shape
                new_weights = np.random.uniform(low=0, high=s.max_weight, size=shape)

                individual.append(new_weights)
            population.append(individual)

        return population

    def calculate_fitness(s, individual):
        total_reward = 0

        for _ in range(s.episole):
            state = s.env.initialize()
            state = s.env.preprogess(state)

            done = False

            while not done:
                action = s.model.act(state)
                next_state, reward, done, info = s.env.step(action)
                state = s.env.preprogess(next_state)
                # s.env.render()

                total_reward += reward

        return total_reward

    # Select the filter_size best individuals baseed on their fitness
    def select_best(s, population, fitness, filter_size):
        # Decending sort and return sorted index array instead of sorted value
        # arrays
        best_indices = np.argsort(fitness)[-filter_size:]

        return [population[i] for i in best_indices]

    # Apply crossover and mutation to create next generation
    def breed(s, parents):
        children = []

        while len(children) < s.population_size // 2:
            # Randomly pick 2 parents
            parent1, parent2 = random.sample(parents, 2)

            parent1 = np.array(parent1, dtype=object)
            parent2 = np.array(parent2, dtype=object)

            # Apply crossover (fusion) on 2 parent to create an offspring parent
            # (child)
            child = s.crossover(parent1, parent2)
            # Mutate (randomize) the child so the model can explore more
            child = s.mutate(child)
            children.append(child)

        return children

    # Apply a crossover operation to combine the genes of the parents
    def crossover(s, parent1, parent2):
        # prob = random.uniform(0, 1)
        # offspring = prob*parent1 + (1 - prob)*parent2
        prob = random.randint(0, len(parent1) - 1)

        offspring = np.concatenate((parent1[:prob], parent2[prob:]))

        return offspring

    def mutate(s, individual, mutation_rate=0.1, mutation_std=0.1):
        # Apply a mutation operation to randomly change the genes of the individual
        # if random.random() < mutation_rate:
        # 	if random.random() < 0.5:
        # 		individual += np.random.randn() * mutation_std
        # 	else:
        # 		individual -= np.random.randn() * mutation_std

        if random.random() < mutation_rate:
            if random.random() < 0.5:
                individual += np.random.normal(0, mutation_std, individual.shape)
            else:
                individual += np.random.normal(0, mutation_std, individual.shape)

        return individual
