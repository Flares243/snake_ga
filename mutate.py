import copy
import numpy as np
import torch

from torch.nn import Module


def mutate(model: Module, mu: float, sigma: float) -> Module:
    # mu - mutation rate. % of gene to be modified
    # sigma - step size of mutation
    # mutation = original gene + (step size * random number)
    y = copy.deepcopy(model.state_dict())

    for name, param in model.named_parameters():
        if param.requires_grad:
            # array of True and False, indicating the mutation position
            flag = np.random.rand(*param.shape) <= mu
            index = np.argwhere(flag)
            mutation = sigma * torch.randn(len(index))
            # print(y[name].shape)
            for i in range(len(index)):
                y[name][*index[i]] += mutation[i]

    model.load_state_dict(y)

    return model
