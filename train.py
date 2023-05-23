from enviroment import SnakeEnviroment
from generic_algo import GenericAlgoOptimizer
from misc import PARENTS_SIZE, loadModel, saveModel
from model import SnakeNeural
from selection import elitism_selection


def save_model(population, fitness):
    response = ""
    while response not in ["y", "n"]:
        response = input("Save model? y/n")

    if response == "y":
        best_model, _ = elitism_selection(population, fitness, PARENTS_SIZE)

        for index in range(len(best_model)):
            saveModel(best_model[index], "{}{}.pt".format(save_file_name, index))


if __name__ == "__main__":
    save_file_name = "fit_func_model"

    env = SnakeEnviroment()
    snake = SnakeNeural()

    loaded_models = []

    response = ""
    while response not in ["y", "n"]:
        response = input("Load model? y/n")

    if response == "y":
        for index in range(PARENTS_SIZE):
            newModel = SnakeNeural()
            loadModel(newModel, "{}{}.pt".format(save_file_name, index))
            loaded_models.append(newModel)

    try:
        generic = GenericAlgoOptimizer(env, snake, loaded_models)
        generic.optimize()

        save_model(generic.population, generic.fitness)
    except:
        save_model(generic.population, generic.fitness)
