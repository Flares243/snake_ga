import os

from enum import Enum
from torch import save, load
from torch.nn import Module


VELOCITY = 1
POPULATION_SIZE = 100
SELECTED_SIZE = 10
MUTATION_RATE = 0.01
MUTATION_STD = 0.001
CROSSOVER_RATE = 0.1


class Slope:
    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run


class Direction(Enum):
    LEFT = [-VELOCITY, 0]
    UP = [0, -VELOCITY]
    RIGHT = [VELOCITY, 0]
    DOWN = [0, VELOCITY]


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
