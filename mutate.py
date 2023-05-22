import copy
import random
import numpy as np
import torch

from torch.nn import Module


def mutate(model: Module, mu: float, sigma: float) -> Module:
    # mu - mutation rate. % of gene to be modified
    # sigma - step size of mutation
    # mutation = original gene + (step size * random number)
    y = copy.deepcopy(model.state_dict())

    for key, params in model.named_parameters():
        if params.requires_grad:
            # array of True and False, indicating the mutation position
            flag = np.random.rand(*params.shape) <= mu
            index = np.argwhere(flag)
            mutation = sigma * torch.randn(len(index))

            for i in range(len(index)):
                y[key][*index[i]] += mutation[i]

    model.load_state_dict(y)

    return model


# def mutate(model: Module) -> Module:
#     for key, params in model.named_parameters():
#         params_tensor = params.data

#         try:
#             iter(params_tensor)
#             for value in range(len(params_tensor)):
#         except:
