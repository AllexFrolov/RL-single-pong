import numpy as np
import cv2
import random
import time


class Canvas(object):
    def __init__(self, width=160, height=210):
        self.color = (255, 255, 255)
        self.width = width
        self.height = height
        self.c = np.zeros((self.height, self.width, 3), np.uint8)
        self.c += np.asarray(self.color, dtype=np.uint8)

    def reset(self):
        self.c = np.zeros((self.height, self.width, 3), np.uint8)
        self.c += np.asarray(self.color, dtype=np.uint8)

    def shape(self):
        return self.c.shape


class Score:
    def __init__(self):
        self.score = 0

    def reset(self):
        self.score = 0

    def hit(self):
        self.score += 1


class Paddle:
    def __init__(self, canvas, score):
        self.color = (20, 20, 20)
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

        elif self.pos['y2'] >= self.canvas.shape()[0] and self.y > 0:
            self.y = 0

        self.move(0, self.y)


class Ball:
    def __init__(self, canvas, paddle, score):
        self.canvas = canvas
        self.paddle = paddle
        self.score = score
        self.pos = {'x': 0, 'y': 0}
        self.radius = 7
        self.color = (100, 100, 100)
        self.hitting_recently = False
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
        if self.paddle.pos['y1'] <= self.pos['y'] + 1 and \
                self.pos['y'] - 1 <= self.paddle.pos['y2'] and \
                not self.hitting_recently:
            if self.paddle.pos['x1'] <= self.pos['x'] + self.radius <= self.paddle.pos['x2']:
                self.score.hit()
                self.speed += self.acceleration
                self.hitting_recently = True
                return True
        return False

    def step(self):
        self.move(self.x, self.y)

        if self.pos['y'] - self.radius <= 0:
            self.y = self.speed
            self.hitting_recently = False
        if self.pos['y'] + self.radius >= self.canvas.shape()[0]:
            self.y = -self.speed
            self.hitting_recently = False

        if self.hit_paddle():
            self.x = -self.speed
        if self.pos['x'] - self.radius <= 0:
            self.x = self.speed
            self.hitting_recently = False
        if self.pos['x'] + self.radius >= self.canvas.shape()[1]:
            self.hit_right = True
            self.hitting_recently = False
            self.score.hit()


class Game:
    def __init__(self, draw=False, video=False):
        self.video = video
        if video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_out = cv2.VideoWriter('output.avi', fourcc, 60, (160, 210))
        self.draw = draw
        self.canvas = Canvas()
        self.scores = np.zeros(2)
        self.states = np.zeros((self.canvas.shape()[0], self.canvas.shape()[1], 3), np.uint8)
        self.score = Score()
        self.paddle = Paddle(self.canvas, self.score)
        self.ball = Ball(self.canvas, self.paddle, self.score)

    def reset(self, reset_score=True):
        self.states = np.zeros((self.canvas.shape()[0], self.canvas.shape()[1], 3), np.uint8)
        self.canvas.reset()
        if reset_score:
            self.score.reset()
            self.scores = np.zeros(2)
        self.paddle.reset()
        self.ball.reset()
        return self.get_state()

    def write_video(self):
        self.video_out.write(self.get_state())

    def get_state(self):
        return self.canvas.c

    @staticmethod
    def distance(coord1, coord2):
        return ((coord1[1] - coord2[1]) ** 2) ** 0.5

    def get_ball_distance(self):
        x_pad_cent = (self.paddle.pos['x1'] + self.paddle.pos['x2']) / 2
        y_pad_cent = (self.paddle.pos['y1'] + self.paddle.pos['y2']) / 2
        x_ball = self.ball.pos['x']
        y_ball = self.ball.pos['y']
        return - self.distance((x_pad_cent, y_pad_cent), (x_ball, y_ball)) / 188

    def delta_score(self):
        self.scores[0] = self.scores[1]
        self.scores[1] = self.score.score
        return self.scores[1] - self.scores[0]

    def step(self, action):
        state = self.get_state()
        ds = self.delta_score()
        if self.video:
            self.write_video()
        if self.score.score == 10:
            done = True
        else:
            done = False

        if not self.ball.hit_right:
            self.paddle.step(action)
            self.ball.step()
            if self.draw:
                cv2.imshow('hi', self.canvas.c)
                cv2.waitKey(10)
                time.sleep(0.0001)
            return state, ds, done, state
        else:
            if not done:
                state = self.reset(False)
            return state, -ds, done, state

    def stop(self):
        try:
            self.video_out.release()
            cv2.destroyWindow('hi')
        except:
            pass


def main():
    game = Game(True)
    for _ in range(2):
        done = False
        rewards = 0
        _ = game.reset()
        while not done:
            state, reward, done, score = game.step(random.randint(0, 2))
            if reward != 0:
                print(f'reward: {reward}, score: {score}')
            rewards += reward

        print(rewards)
    game.stop()


if __name__ == '__main__':
    main()
