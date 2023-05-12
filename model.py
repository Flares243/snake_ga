import torch
from torch.nn import Flatten, Sequential, Linear, ReLU, Module


class SnakeNeural(Module):
    def __init__(s) -> None:
        super(SnakeNeural, s).__init__()
        s.flatten = Flatten()
        s.model = Sequential(
            Linear(32, 78),
            ReLU(),
            Linear(78, 56),
            ReLU(),
            Linear(56, 4),
        )

    def forward(s, x):
        x = s.preprocces(x)
        pred = s.model(x)
        return pred

    def preprocces(s, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.reshape(1, -1)
        x = s.flatten(x)
        return x
