import random

from torch.nn import Module
from typing import Tuple


def single_point_crossover(p1: Module, p2: Module) -> Tuple[Module, Module]:
    # Get the state dicts of the two parents
    parent1_state_dict = p1.state_dict()
    parent2_state_dict = p2.state_dict()

    # Choose a random crossover point in the range of valid layers
    number_of_layers = len(parent1_state_dict.keys())
    crossover_point = random.randrange(0, number_of_layers, step=2)

    # Swap weights between parent models
    child1_state_dict = {
        key: parent1_state_dict[key] if i < crossover_point else parent2_state_dict[key]
        for i, key in enumerate(parent1_state_dict.keys())
    }
    child2_state_dict = {
        key: parent2_state_dict[key] if i < crossover_point else parent1_state_dict[key]
        for i, key in enumerate(parent2_state_dict.keys())
    }

    # Create the child models with swapped weights
    child1 = type(p1)()
    child2 = type(p2)()
    child1.load_state_dict(child1_state_dict)
    child2.load_state_dict(child2_state_dict)

    return child1, child2


# def uniform_binary_crossover(p1: SnakeNeural, p2: SnakeNeural):
#     offspring1 = p1.model.copy()
#     offspring2 = p2.model.copy()

#     mask = np.random.uniform(0, 1, size=offspring1.shape)
#     offspring1[mask > 0.5] = p2[mask > 0.5]
#     offspring2[mask > 0.5] = p1[mask > 0.5]

#     return offspring1, offspring2
