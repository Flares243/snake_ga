from enviroment import SnakeEnviroment
from generic_algo import GenericAlgoOptimizer
from misc import loadModel, saveModel
from model import SnakeNeural
from selection import elitism_selection

if __name__ == "__main__":
    save_file_name = "pure_score_model.pt"

    env = SnakeEnviroment()
    snake = SnakeNeural()

    response = ""
    while response not in ["y", "n"]:
        response = input("Load model? y/n")

    if response == "y":
        loadModel(snake, save_file_name)

    try:
        generic = GenericAlgoOptimizer(env, snake)
        best_model = generic.optimize()

        response = ""
        while response not in ["y", "n"]:
            response = input("Save model? y/n")

        if response == "y":
            saveModel(best_model, file_name=save_file_name)
    except:
        response = ""
        while response not in ["y", "n"]:
            response = input("Save model? y/n")

        if response == "y":
            best_model, _ = elitism_selection(generic.population, generic.fitness, 1)[0]
            saveModel(best_model, file_name=save_file_name)
