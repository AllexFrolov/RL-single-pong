import numpy as np
import cv2
import random
import time


class Canvas(object):
    def __init__(self, width=500, height=400, color=255):
        self.color = color
        self.width = width
        self.height = height
        self.c = np.zeros((self.height, self.width, 1), np.uint8)
        self.c += self.color

    def reset(self):
        self.c = np.zeros((self.height, self.width, 1), np.uint8)
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
    def __init__(self, canvas, score):
        self.canvas = canvas
        self.score = score
        self.pos = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 10}
        start = [0, 100, 250, 400]
        starting_point_x = random.choice(start)
        starting_point_y = 300
        self.move(starting_point_x, starting_point_y)
        self.x = 0

    def reset(self):
        start = [0, 100, 250, 400]
        starting_point_x = random.choice(start)
        starting_point_y = 300
        self.pos = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 10}
        self.move(starting_point_x, starting_point_y)
        self.x = 0

    def move(self, x, y):
        cv2.rectangle(self.canvas.c, (self.pos['x1'], self.pos['y1']),
                      (self.pos['x2'], self.pos['y2']),
                      255, thickness=-1)
        self.pos = {'x1': self.pos['x1'] + x,
                    'y1': self.pos['y1'] + y,
                    'x2': self.pos['x2'] + x,
                    'y2': self.pos['y2'] + y}
        cv2.rectangle(self.canvas.c, (self.pos['x1'], self.pos['y1']),
                      (self.pos['x2'], self.pos['y2']),
                      100, thickness=-1)

    def turn(self, action):
        if action == 2:
            self.x = 20
        elif action == 0:
            self.x = -20
        else:
            self.x = 0

    def step(self, action):
        self.turn(action)
        if self.pos['x1'] <= 0 and self.x < 0:
            self.x = 0
            self.score.hit_wall()

        elif self.pos['x2'] >= self.canvas.shape()[1] and self.x > 0:
            self.x = 0
            self.score.hit_wall()

        self.move(self.x, 0)

    def drawing(self):
        pass


class Ball:
    def __init__(self, canvas, paddle, score):
        self.canvas = canvas
        self.paddle = paddle
        self.score = score
        self.pos = {'x': 0, 'y': 0}
        starting_point_x = 250
        starting_point_y = 245
        self.radius = 15
        self.move(starting_point_x, starting_point_y)
        self.x = random.randint(-10, 10)
        self.y = -10
        self.hit_bottom = False

    def reset(self):
        self.pos = {'x': 0, 'y': 0}
        starting_point_x = 250
        starting_point_y = 245
        self.move(starting_point_x, starting_point_y)
        self.x = random.randint(-10, 10)
        self.y = -10
        self.hit_bottom = False

    def move(self, x, y):
        cv2.circle(self.canvas.c, center=(self.pos['x'], self.pos['y']),
                   radius=self.radius, color=255, thickness=-1)
        self.pos = {'x': self.pos['x'] + x,
                    'y': self.pos['y'] + y}
        cv2.circle(self.canvas.c, center=(self.pos['x'], self.pos['y']),
                   radius=self.radius, color=200, thickness=-1)

    def hit_paddle(self):
        if self.paddle.pos['x1'] <= self.pos['x'] <= self.paddle.pos['x2']:
            if self.paddle.pos['y1'] <= self.pos['y'] + self.radius <= self.paddle.pos['y2']:
                self.score.hit()
                return True
        return False

    def step(self):
        self.move(self.x, self.y)

        if self.pos['y'] - self.radius <= 0:
            self.y = 10
        if self.pos['y'] + self.radius >= self.canvas.shape()[0]:
            self.hit_bottom = True

        if self.hit_paddle():
            self.y = -10
        if self.pos['x'] - self.radius <= 0:
            self.x = 10
        if self.pos['x'] + self.radius >= self.canvas.shape()[1]:
            self.x = -10


class Game:
    def __init__(self, draw=False):
        self.draw = draw
        self.canvas = Canvas()
        self.scores = np.zeros(2)
        self.states = np.zeros((3, 400, 500, 3), np.uint8)
        self.score = Score()
        self.paddle = Paddle(self.canvas, self.score)
        self.ball = Ball(self.canvas, self.paddle, self.score)

    def reset(self):
        self.scores = np.zeros(2)
        self.states = np.zeros((3, 400, 500, 3), np.uint8)
        self.canvas.reset()
        self.score.reset()
        self.paddle.reset()
        self.ball.reset()
        return self.canvas.c

    def get_state(self):
        for ind in range(self.states.shape[0]-1):
            self.states[ind] = - np.abs(self.states[ind+1])
        self.states[-1] = self.canvas.c
        return self.states.sum(axis=0, dtype=np.uint8)

    def delta_score(self):
        self.scores[0] = self.scores[1]
        self.scores[1] = self.score.score
        return self.scores[1] - self.scores[0]

    def step(self, action):
        if not self.ball.hit_bottom:
            self.paddle.step(action)
            self.ball.step()
            if self.draw:
                cv2.imshow('hi', self.canvas.c)
                cv2.waitKey(10)
                # time.sleep(0.01)
            if self.score.score == 5:
                return self.get_state(), 3, True, self.canvas.c

            return self.get_state(), self.delta_score(), False, self.canvas.c
        else:
            return self.get_state(), -1, True, self.canvas.c

    def stop(self):
        cv2.destroyWindow('hi')

def main():
    game = Game(True)
    done = False
    rewards = 0
    while not done:
        state, reward, done, _ = game.step(random.randint(0, 2))
        if reward > 0:
            print(reward)
        rewards += reward

    print(rewards)
    game.stop()


if __name__ == '__main__':
    main()
