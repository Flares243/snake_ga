from model import SnakeNeural
from enviroment import SnakeEnviroment

from generic_algo import GenericAlgoOptimizer
from misc import saveModel, loadModel
from selection import elitism_selection

if __name__ == "__main__":
    save_file_name = "pure_score_model.pth"

    env = SnakeEnviroment()
    snake = SnakeNeural()

    for key, params in snake.named_parameters():
        print("key")
        print(key)
        print("params")
        print(iter(params.data))
        print(params.data)

    response = ""
    while response not in ["y", "n"]:
        response = input("Load model? y/n")

    if response == "y":
        loadModel(snake, save_file_name)

    generic = GenericAlgoOptimizer(env, snake)

    try:
        generic.optimize()
    except:
        response = ""
        while response not in ["y", "n"]:
            response = input("Save model? y/n")

        if response == "y":
            best_model = elitism_selection(generic.population, generic.fitness, 1)[0]
            saveModel(best_model, file_name=save_file_name)
