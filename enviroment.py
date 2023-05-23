import random

import numpy as np
import pygame

from misc import Direction, Slope

pygame.init()
pygame.display.set_caption("Snake")

clock = pygame.time.Clock()

font = pygame.font.Font(None, 36)
food_color = pygame.Color(255, 0, 0)
snake_color = pygame.Color(255, 255, 255)
score_color = pygame.Color(255, 255, 255)
background_color = pygame.Color(0, 0, 0)


class SnakeEnviroment:
    def __init__(self, map_size=10) -> None:
        self.fps = 60
        self.game_speed = 1
        self.square_size = 30
        self.map_size = map_size

        self.window_size = (
            self.map_size * self.square_size,
            self.map_size * self.square_size,
        )

        self.screen = pygame.display.set_mode(self.window_size)

        self.initialize()

    def initialize(s) -> None:
        s.reward = 0
        s.score = 0
        s.frames = 0
        s.step_count = 0
        s.game_over = False

        # s.snake_direction = Direction.DOWN
        s.snake_direction = random.choice(list(Direction))
        s.tail_direction = s.snake_direction

        # init_snake_pos = [0, 9]
        init_snake_pos = [s.map_size // 2, s.map_size // 2]

        s.snake = [
            init_snake_pos,
            [
                init_snake_pos[i] + (-1 * s.snake_direction.value[i])
                for i in range(len(init_snake_pos))
            ],
        ]

        s.food_spawn = False
        s.generate_food()

    def generate_food(s) -> None:
        if not s.food_spawn:
            s.food_position = random.choices(range(0, s.map_size), k=2)

            if s.food_position in s.snake:
                s.generate_food()
                return

            s.food_spawn = True

    def check_eat_food(s) -> None:
        if np.array_equal(s.snake[0], s.food_position):
            s.reward = 1
            s.score += 1
            s.step_count = 0
            s.food_spawn = False
        else:
            s.snake.pop()

    def check_game_over(s) -> None:
        if s.step_count > len(s.snake) * 100:
            s.reward = -1
            s.game_over = True
            s.score = 0
            return

        if s.is_wall_collide(s.snake[0]) or s.is_body_collide(s.snake[0]):
            s.reward = -1
            s.game_over = True

    def update_snake_direction(s, action: Direction) -> None:
        if action == Direction.LEFT and s.snake_direction != Direction.RIGHT:
            s.snake_direction = Direction.LEFT
        elif action == Direction.UP and s.snake_direction != Direction.DOWN:
            s.snake_direction = Direction.UP
        elif action == Direction.RIGHT and s.snake_direction != Direction.LEFT:
            s.snake_direction = Direction.RIGHT
        elif action == Direction.DOWN and s.snake_direction != Direction.UP:
            s.snake_direction = Direction.DOWN

    def update_tail_direction(s) -> None:
        snake_tail = s.snake[-1]
        next_snake_tail = s.snake[-2]

        if s.tail_direction != Direction.RIGHT and snake_tail[0] > next_snake_tail[0]:
            s.tail_direction = Direction.LEFT
        elif s.tail_direction != Direction.UP and snake_tail[1] > next_snake_tail[1]:
            s.tail_direction = Direction.UP
        elif s.tail_direction != Direction.RIGHT and snake_tail[0] < next_snake_tail[0]:
            s.tail_direction = Direction.RIGHT
        elif s.tail_direction != Direction.DOWN and snake_tail[1] < next_snake_tail[1]:
            s.tail_direction = Direction.DOWN

    def move_snake(s):
        s.snake.insert(
            0,
            [
                s.snake[0][i] + s.snake_direction.value[i]
                for i in range(len(s.snake[0]))
            ],
        )

        # if s.snake_direction == Direction.LEFT:
        #     s.snake[0][0] -= 1
        # elif s.snake_direction == Direction.UP:
        #     s.snake[0][1] -= 1
        # elif s.snake_direction == Direction.RIGHT:
        #     s.snake[0][0] += 1
        # elif s.snake_direction == Direction.DOWN:
        #     s.snake[0][1] += 1

    def step(s, action: Direction) -> None:
        s.step_count += 1
        s.frames += 1

        s.update_snake_direction(action)
        s.update_tail_direction()
        s.move_snake()
        s.check_eat_food()
        s.check_game_over()
        s.generate_food()
        s.render()

    def get_state(s) -> list:
        snake_direction = np.eye(len(Direction))[
            list(Direction).index(s.snake_direction)
        ]

        tail_direction = np.eye(len(Direction))[
            list(Direction).index(s.snake_direction)
        ]

        west = s.look_in_direction(Slope(run=-1, rise=0))
        north = s.look_in_direction(Slope(run=0, rise=-1))
        east = s.look_in_direction(Slope(run=1, rise=0))
        south = s.look_in_direction(Slope(run=0, rise=1))
        north_west = s.look_in_direction(Slope(run=-1, rise=-1))
        north_east = s.look_in_direction(Slope(run=1, rise=-1))
        south_west = s.look_in_direction(Slope(run=-1, rise=1))
        south_east = s.look_in_direction(Slope(run=1, rise=1))

        # print("\n")
        # print("west")
        # print(west)
        # print("north_west")
        # print(north_west)
        # print("north")
        # print(north)
        # print("north_east")
        # print(north_east)
        # print("east")
        # print(east)
        # print("south_east")
        # print(south_east)
        # print("south")
        # print(south)
        # print("south_west")
        # print(south_west)

        obs = np.concatenate(
            [
                snake_direction,
                tail_direction,
                west,
                north_west,
                north,
                north_east,
                east,
                south_east,
                south,
                south_west,
            ],
            dtype=np.float32,
        )

        return obs.tolist()

    def get_info(s) -> tuple:
        return (s.frames, s.reward, s.score, s.game_over)

    def render(s) -> None:
        s.screen.fill(background_color)

        for snake_pos in s.snake:
            pygame.draw.rect(
                s.screen,
                snake_color,
                pygame.Rect(
                    snake_pos[0] * s.square_size,
                    snake_pos[1] * s.square_size,
                    s.square_size,
                    s.square_size,
                ),
            )

        pygame.draw.rect(
            s.screen,
            food_color,
            pygame.Rect(
                s.food_position[0] * s.square_size,
                s.food_position[1] * s.square_size,
                s.square_size,
                s.square_size,
            ),
        )

        score_font = font.render(str(s.score), True, score_color)
        s.screen.blit(score_font, (5, 10))

        pygame.display.update()
        clock.tick(s.fps * s.game_speed)

    def look_in_direction(s, slope: Slope) -> list[bool]:
        position = s.snake[0].copy()

        has_body = False
        has_food = False
        has_space = False

        # distance = 1
        # total_distance = 0
        # distance_to_wall = None
        # distance_to_body = np.inf
        # distance_to_food = np.inf

        position[0] += slope.run
        position[1] += slope.rise
        # total_distance += distance

        if not s.is_body_collide(position) and not s.is_wall_collide(position):
            has_space = True

        while not s.is_wall_collide(position):
            if s.is_body_collide(position):
                has_body = True
                # distance_to_body = total_distance
            if s.is_food_collide(position):
                has_food = True
                # distance_to_food = total_distance

            position[0] += slope.run
            position[1] += slope.rise
            # total_distance += distance

        # distance_to_wall = (total_distance - 1) / s.map_size
        # distance_to_body = distance_to_body / s.map_size if distance_to_body != np.inf else 1
        # distance_to_food = distance_to_food / s.map_size if distance_to_food != np.inf else 1

        return [has_space, has_body, has_food]

    def is_body_collide(s, position: list) -> bool:
        return position in s.snake[1:]

    def is_food_collide(s, position: list) -> bool:
        return np.array_equal(position, s.food_position)

    def is_wall_collide(s, position: list) -> bool:
        return (
            position[0] < 0
            or position[0] > s.map_size - 1
            or position[1] < 0
            or position[1] > s.map_size - 1
        )

    def calc_distance(x1, x2, y1, y2) -> float:
        diff_x = float(abs(x2 - x1))
        diff_y = float(abs(y2 - y1))
        dist = ((diff_x * diff_x) + (diff_y * diff_y)) ** 0.5
        return dist
        return dist
