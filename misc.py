import os
from enum import Enum
from typing import Any

from torch import load, save
from torch.nn import Module

VELOCITY = 1
POPULATION_SIZE = 100
PARENTS_SIZE = 10
MUTATION_RATE = 0.05
MUTATION_STD = 10
CROSSOVER_RATE = 0.2


class Slope:
    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run


class Direction(Enum):
    LEFT = [-VELOCITY, 0]
    UP = [0, -VELOCITY]
    RIGHT = [VELOCITY, 0]
    DOWN = [0, VELOCITY]


def one_hot_encode(input_list: list[Any]) -> list[Any]:
    unique_values = list(set(input_list))
    num_unique = len(unique_values)
    encoding = []

    for value in input_list:
        one_hot = [0] * num_unique
        index = unique_values.index(value)
        one_hot[index] = 1
        encoding.append(one_hot)

    return encoding


def saveModel(model: Module, file_name="model.pt"):
    folder_path = "./model"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    save(model.state_dict(), file_path)


def loadModel(model: Module, file_name="model.pt"):
    folder_path = "./model"
    file_path = os.path.join(folder_path, file_name)
    saved_state_dict = load(file_path)
    model.load_state_dict(saved_state_dict)
