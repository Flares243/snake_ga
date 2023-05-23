from copy import deepcopy
from random import randint, random

import numpy as np
import torch
from torch.nn import Module

from misc import MUTATION_RATE, MUTATION_STD


def mutate(model: Module, mu: float, sigma: float) -> Module:
    # mu - mutation rate. % of gene to be modified
    # sigma - step size of mutation
    # mutation = original gene + (step size * random number)
    y = deepcopy(model.state_dict())

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


def mutation(model: Module) -> Module:
    newModel = deepcopy(model)

    for key, params in newModel.named_parameters():
        param_tensors = params.data

        for tensor_index in range(len(param_tensors)):
            try:
                for value_index in range(len(param_tensors[tensor_index])):
                    probability = random()
                    change = randint(-MUTATION_STD, MUTATION_STD) / 1000

                    if probability <= MUTATION_RATE:
                        param_tensors[tensor_index][value_index] += (
                            param_tensors[tensor_index][value_index] * change
                        )
            except TypeError:
                probability = random()
                change = randint(-MUTATION_STD, MUTATION_STD) / 1000

                if probability <= MUTATION_RATE:
                    param_tensors[tensor_index] += param_tensors[tensor_index] * change

        newModel.state_dict()[key] = param_tensors

    return newModel
