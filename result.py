import torch

from enviroment import SnakeEnviroment
from misc import loadModel
from model import SnakeNeural

env = SnakeEnviroment()
snaky = SnakeNeural()

loadModel(snaky, file_name="pure_score_model.pt")

snaky.eval()

done = False
state = env.get_state()

with torch.no_grad():
    while not done:
        pred = snaky(state)
        action = torch.argmax(pred).item()

        env.step(action)

        env.render()

        state = env.get_state()
        _, __, done = env.get_info()
        _, __, done = env.get_info()
