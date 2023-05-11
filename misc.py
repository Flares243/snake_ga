import os

from enum import Enum
from torch import save, load
from torch.nn import Module


class Slope:
    __slots__ = ("rise", "run")

    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


def saveModel(model: Module, file_name="model.pth"):
    folder_path = "./model"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    save(model.state_dict(), file_path)


def loadModel(model: Module, file_name="model.pth"):
    folder_path = "./model"
    file_path = os.path.join(folder_path, file_name)
    saved_state_dict = load(file_path)
    model.load_state_dict(saved_state_dict)
