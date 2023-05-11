from model import SnakeNeural
from enviroment import SnakeEnviroment
from generic_algo import GenericAlgoOptimizer
from misc import saveModel

env = SnakeEnviroment()
snake = SnakeNeural()

generic = GenericAlgoOptimizer(env, snake)

best_model = generic.optimize()

saveModel(best_model)
