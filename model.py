from torch import optim, nn, tensor, float32, max
from torch.nn import Flatten, Sequential, Linear, Module, Sigmoid


class SnakeNeural(Module):
    def __init__(s) -> None:
        super(SnakeNeural, s).__init__()
        s.flatten = Flatten()
        s.model = Sequential(
            Linear(36, 78),
            Linear(78, 4),
            Sigmoid(),
        )

        # s.lr = 1e-1
        # s.discount = 0.3
        # s.optimizer = optim.Adam(s.model.parameters(), lr=s.lr)
        # s.loss_fn = nn.MSELoss()

    def forward(s, x: list[int]) -> int:
        x = tensor(x)
        x = x.reshape(1, -1)
        x = s.flatten(x)

        pred = s.model(x).tolist()

        return pred.index(max(pred))

    # def traine(s, state, action, reward, next_state, game_over):
    #     pred = s.forward(state)
    #     target = pred.clone()

    #     next_q = reward

    #     if not game_over:
    #         next_pred = s.forward(next_state)
    #         next_q = reward + s.discount * max(next_pred).item()

    #     target[0][action] = next_q

    #     s.optimizer.zero_grad()
    #     loss = s.loss_fn(target, pred)
    #     loss.backward()
    #     s.optimizer.step()
