import random
import pygame
import numpy as np

from misc import Slope, Direction

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
        self.fps = 5
        self.game_speed = 2

        self.map_size = map_size
        self.square_size = 30

        self.window_size = (
            self.map_size * self.square_size,
            self.map_size * self.square_size,
        )

        self.screen = pygame.display.set_mode(self.window_size)

        self.initialize()

    def initialize(s):
        s.reward = 0
        s.score = 0
        s.step_count = 0
        s.game_over = False

        s.snake_direction = Direction.RIGHT.value
        s.tail_direction = Direction.RIGHT.value

        init_snake_pos = s.map_size // 2
        s.snake = [
            [init_snake_pos, init_snake_pos],
            [init_snake_pos - 1, init_snake_pos],
        ]

        s.food_spawn = False
        s.generate_food()

    def generate_food(s):
        if not s.food_spawn:
            s.food_position = [
                random.randrange(0, s.map_size),
                random.randrange(0, s.map_size),
            ]

            if s.food_position in s.snake:
                s.generate_food()
                return

            s.food_spawn = True

    def check_eat_food(s):
        if np.array_equal(s.snake[0], s.food_position):
            s.food_spawn = False
            s.reward = 10
            s.step_count = 0
            s.score += 1
        else:
            s.snake.pop()

    def check_game_over(s):
        if s.step_count > (len(s.snake) * 100):
            s.game_over = True
            s.reward = -20
            return
        if s.is_wall_collide(s.snake[0]):
            s.game_over = True
            s.reward = -10
            return

        for snake_body in s.snake[1:]:
            if np.array_equal(s.snake[0], snake_body):
                s.game_over = True
                s.reward = -10
                break

    def update_snake_direction(s, action):
        if (
            action == Direction.LEFT.value
            and s.snake_direction != Direction.RIGHT.value
        ):
            s.snake_direction = Direction.LEFT.value
        elif action == Direction.UP.value and s.snake_direction != Direction.DOWN.value:
            s.snake_direction = Direction.UP.value
        elif (
            action == Direction.RIGHT.value
            and s.snake_direction != Direction.LEFT.value
        ):
            s.snake_direction = Direction.RIGHT.value
        elif action == Direction.DOWN.value and s.snake_direction != Direction.UP.value:
            s.snake_direction = Direction.DOWN.value

    def update_tail_direction(s):
        snake_tail = s.snake[-1]
        next_snake_tail = s.snake[-2]

        if (
            s.tail_direction != Direction.RIGHT.value
            and snake_tail[0] > next_snake_tail[0]
        ):
            s.tail_direction = Direction.LEFT.value
        elif (
            s.tail_direction != Direction.UP.value
            and snake_tail[1] > next_snake_tail[1]
        ):
            s.tail_direction = Direction.UP.value
        elif (
            s.tail_direction != Direction.RIGHT.value
            and snake_tail[0] < next_snake_tail[0]
        ):
            s.tail_direction = Direction.RIGHT.value
        elif (
            s.tail_direction != Direction.DOWN.value
            and snake_tail[1] < next_snake_tail[1]
        ):
            s.tail_direction = Direction.DOWN.value

    def move_snake(s):
        s.snake.insert(0, list(s.snake[0]))

        if s.snake_direction == Direction.LEFT.value:
            s.snake[0][0] -= 1
        elif s.snake_direction == Direction.UP.value:
            s.snake[0][1] -= 1
        elif s.snake_direction == Direction.RIGHT.value:
            s.snake[0][0] += 1
        elif s.snake_direction == Direction.DOWN.value:
            s.snake[0][1] += 1

    def step(s, action):
        s.step_count += 1

        s.update_snake_direction(action)
        s.update_tail_direction()
        s.move_snake()
        s.check_eat_food()
        s.check_game_over()
        s.generate_food()

    def get_obs(s):
        snake_direction = np.eye(len(Direction))[s.snake_direction]
        tail_direction = np.eye(len(Direction))[s.tail_direction]

        west = s.look_in_direction(Slope(run=-1, rise=0))
        north = s.look_in_direction(Slope(run=0, rise=-1))
        east = s.look_in_direction(Slope(run=1, rise=0))
        south = s.look_in_direction(Slope(run=0, rise=1))
        north_west = s.look_in_direction(Slope(run=-1, rise=-1))
        north_east = s.look_in_direction(Slope(run=1, rise=-1))
        south_west = s.look_in_direction(Slope(run=-1, rise=1))
        south_east = s.look_in_direction(Slope(run=1, rise=1))

        print("\n")
        print(west)
        print(north)
        print(east)
        print(south)
        print(north_west)
        print(north_east)
        print(south_west)
        print(south_east)

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
            ]
        )

        return obs

    def get_info(s):
        return s.reward, s.score, s.game_over

    def render(s):
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

    def look_in_direction(s, slope: Slope):
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
            if not has_body and not has_food:
                if s.is_body_collide(position):
                    has_body = True
                    # distance_to_body = total_distance
                elif s.is_food_collide(position):
                    has_food = True
                    # distance_to_food = total_distance

            position[0] += slope.run
            position[1] += slope.rise
            # total_distance += distance

        # distance_to_wall = (total_distance - 1) / s.map_size
        # distance_to_body = distance_to_body / s.map_size if distance_to_body != np.inf else 1
        # distance_to_food = distance_to_food / s.map_size if distance_to_food != np.inf else 1

        return [has_space, has_food, has_body]

    def is_body_collide(s, position):
        if isinstance(position, np.ndarray):
            return position in np.array(s.snake[1:])
        else:
            return position in s.snake[1:]

    def is_food_collide(s, position):
        return position[0] == s.food_position[0] and position[1] == s.food_position[1]

    def is_wall_collide(s, position):
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
