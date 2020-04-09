import numpy as np
import cv2
import random
import time


class Canvas(object):
    def __init__(self, width=160, height=210, color=255):
        self.color = color
        self.width = width
        self.height = height
        self.c = np.zeros((self.height, self.width), np.uint8)
        self.c += self.color

    def reset(self):
        self.c = np.zeros((self.height, self.width), np.uint8)
        self.c += self.color

    def shape(self):
        return self.c.shape


class Score:
    def __init__(self):
        self.score = 0

    def reset(self):
        self.score = 0

    def hit(self):
        self.score += 1

    def hit_wall(self):
        self.score -= 0.0


class Paddle:
    def __init__(self, canvas, score, color=20):
        self.color = color
        self.canvas = canvas
        self.score = score
        self.reset()

    def reset(self):
        self.pos = {'x1': 0, 'y1': 0, 'x2': 5, 'y2': 30}
        starting_point_y = random.randrange(0, 180, 1)
        starting_point_x = 140
        self.move(starting_point_x, starting_point_y)
        self.y = 0

    def move(self, x, y):
        cv2.rectangle(self.canvas.c, (self.pos['x1'], self.pos['y1']),
                      (self.pos['x2'], self.pos['y2']),
                      self.canvas.color, thickness=-1)
        self.pos = {'x1': self.pos['x1'] + x,
                    'y1': self.pos['y1'] + y,
                    'x2': self.pos['x2'] + x,
                    'y2': self.pos['y2'] + y}
        cv2.rectangle(self.canvas.c, (self.pos['x1'], self.pos['y1']),
                      (self.pos['x2'], self.pos['y2']),
                      self.color, thickness=-1)

    def turn(self, action):
        if action == 2:
            self.y = 4
        elif action == 0:
            self.y = -4
        else:
            self.y = 0

    def step(self, action):
        self.turn(action)
        if self.pos['y1'] <= 0 and self.y < 0:
            self.y = 0
            self.score.hit_wall()

        elif self.pos['y2'] >= self.canvas.shape()[0] and self.y > 0:
            self.y = 0
            self.score.hit_wall()

        self.move(0, self.y)


class Ball:
    def __init__(self, canvas, paddle, score, color=100):
        self.canvas = canvas
        self.paddle = paddle
        self.score = score
        self.pos = {'x': 0, 'y': 0}
        self.radius = 7
        self.color = color
        self.reset()

    def reset(self):
        self.pos = {'x': 0, 'y': 0}
        starting_point_x = int(self.canvas.shape()[1] / 2)
        starting_point_y = int(self.canvas.shape()[0] / 2)
        self.move(starting_point_x, starting_point_y)
        self.x = -1
        self.y = random.randint(-2, 2)
        self.speed = 1
        self.acceleration = 1
        self.hit_right = False

    def move(self, x, y):
        cv2.circle(self.canvas.c, center=(self.pos['x'], self.pos['y']),
                   radius=self.radius, color=self.canvas.color, thickness=-1)
        self.pos = {'x': self.pos['x'] + x,
                    'y': self.pos['y'] + y}
        cv2.circle(self.canvas.c, center=(self.pos['x'], self.pos['y']),
                   radius=self.radius, color=self.color, thickness=-1)

    def hit_paddle(self):
        if self.paddle.pos['y1'] <= self.pos['y'] + 1 and self.pos['y'] - 1 <= self.paddle.pos['y2']:
            if self.paddle.pos['x1'] <= self.pos['x'] + self.radius <= self.paddle.pos['x2'] + 2:
                self.score.hit()
                self.speed += self.acceleration
                return True
        return False

    def step(self):
        self.move(self.x, self.y)

        if self.pos['y'] - self.radius <= 0:
            self.y = self.speed
        if self.pos['y'] + self.radius >= self.canvas.shape()[0]:
            self.y = -self.speed

        if self.hit_paddle():
            self.x = -self.speed
        if self.pos['x'] - self.radius <= 0:
            self.x = self.speed
        if self.pos['x'] + self.radius >= self.canvas.shape()[1]:
            self.hit_right = True


class Game:
    def __init__(self, draw=False):
        self.draw = draw
        self.canvas = Canvas()
        self.scores = np.zeros(2)
        self.states = np.zeros((self.canvas.shape()[0], self.canvas.shape()[1], 4), np.uint8)
        self.score = Score()
        self.paddle = Paddle(self.canvas, self.score)
        self.ball = Ball(self.canvas, self.paddle, self.score)

    def reset(self):
        self.scores = np.zeros(2)
        self.states = np.zeros((self.canvas.shape()[0], self.canvas.shape()[1], 4), np.uint8)
        self.canvas.reset()
        self.score.reset()
        self.paddle.reset()
        self.ball.reset()
        return self.get_state()

    def get_state(self):
        for ind in range(self.states.shape[-1] - 1):
            self.states[..., ind] = self.states[..., ind + 1]
        self.states[..., -1] = self.canvas.c
        return self.states

    def delta_score(self):
        self.scores[0] = self.scores[1]
        self.scores[1] = self.score.score
        return self.scores[1] - self.scores[0]

    def step(self, action):
        if not self.ball.hit_right:
            self.paddle.step(action)
            self.ball.step()
            if self.draw:
                cv2.imshow('hi', self.canvas.c)
                cv2.waitKey(10)
                time.sleep(0.0001)
            if self.score.score == 3:
                return self.get_state(), 1, True, self.canvas.c

            return self.get_state(), self.delta_score(), False, self.canvas.c
        else:
            return self.get_state(), -1, True, self.canvas.c

    @staticmethod
    def stop():
        try:
            cv2.destroyWindow('hi')
        except:
            pass


def main():
    game = Game(True)
    for _ in range(5):
        done = False
        rewards = 0
        _ = game.reset()
        while not done:
            state, reward, done, _ = game.step(random.randint(0, 2))
            if reward > 0:
                print(reward)
            rewards += reward

        print(rewards)
    game.stop()


if __name__ == '__main__':
    main()
