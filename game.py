import random
import time
from tkinter import *
import numpy as np


class Ball:
    def __init__(self, canvas, paddle, score, color, drawing=False):
        self.drawing = drawing
        self.canvas = canvas
        self.paddle = paddle
        self.score = score

        if self.drawing:
            self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
            self.canvas.move(self.id, 245, 100)
            self.canvas_height = self.canvas.winfo_height()
            self.canvas_width = self.canvas.winfo_width()
        else:
            self.canvas_height = self.canvas['height']
            self.canvas_width = self.canvas['width']
        self.pos = [10, 10, 25, 25]
        self.move(245, 100)
        # список возможных направлений для старта
        starts = [-20, -10, 10, 20]
        random.shuffle(starts)
        self.x = starts[0]
        self.y = -20

        self.hit_bottom = False

    def move(self, x, y):
        self.pos = [self.pos[0] + x,
                    self.pos[1] + y,
                    self.pos[2] + x,
                    self.pos[3] + y]

    def reset(self):
        if self.drawing:
            self.canvas.coords(self.id, 10, 10, 25, 25)
            self.canvas.move(self.id, 245, 100)
        self.pos = [10, 10, 25, 25]
        self.move(245, 100)

        starts = [-20, -10, 10, 20]
        random.shuffle(starts)
        self.x = starts[0]
        self.y = -20
        self.hit_bottom = False

    def hit_paddle(self, pos):
        paddle_pos = self.paddle.pos
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if paddle_pos[1] <= pos[3] <= paddle_pos[3]:
                self.score.hit()
                return True
        return False

    def draw(self):
        if self.drawing:
            self.canvas.move(self.id, self.x, self.y)
        self.move(self.x, self.y)
        pos = self.pos

        if pos[1] <= 0:
            self.y = 20
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True

        if self.hit_paddle(pos):
            self.y = -20
        if pos[0] <= 0:
            self.x = 20
        if pos[2] >= self.canvas_width:
            self.x = -20


class Paddle:
    def __init__(self, canvas, score, color, drawing=False):
        self.drawing = drawing
        self.canvas = canvas
        self.score = score
        if self.drawing:
            self.canvas_width = self.canvas.winfo_width()
            self.id = self.canvas.create_rectangle(0, 0, 100, 10, fill=color)
        else:
            self.canvas_width = self.canvas['width']
        self.pos = [0, 0, 100, 10]
        start_1 = [40, 100, 150, 200, 250, 300, 360]
        random.shuffle(start_1)
        self.starting_point_x = start_1[0]
        self.starting_point_y = 300
        if self.drawing:
            self.canvas.move(self.id, self.starting_point_x, self.starting_point_y)
        self.move(self.starting_point_x, self.starting_point_y)
        self.x = 0

    def reset(self):
        start_1 = [40, 60, 90, 120, 150, 180, 200]
        random.shuffle(start_1)
        self.starting_point_x = start_1[0]
        if self.drawing:
            self.canvas.coords(self.id, 0, 0, 100, 10)
            self.canvas.move(self.id, self.starting_point_x, self.starting_point_y)
        self.pos = [0, 0, 100, 10]
        self.move(self.starting_point_x, self.starting_point_y)

        self.x = 0

    def move(self, x, y):
        self.pos = [self.pos[0] + x,
                    self.pos[1] + y,
                    self.pos[2] + x,
                    self.pos[3] + y]

    def turn(self, action):
        if action == 2:
            self.x = 20
        elif action == 0:
            self.x = -20
        else:
            self.x = 0

    def draw(self):
        pos = self.pos
        if pos[0] <= 0 and self.x < 0:
            self.x = 0
            self.score.hit_wall()
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0
            self.score.hit_wall()

        if self.drawing:
            self.canvas.move(self.id, self.x, 0)
        self.move(self.x, 0)


class Score:
    def __init__(self, canvas, color, drawing=False):
        self.drawing = drawing
        self.score = 0
        self.add_score = 1
        self.canvas = canvas
        if self.drawing:
            self.id = canvas.create_text(450, 10, text=self.score,
                                         font=('Courier', 15), fill=color)

    def reset(self):
        self.score = 0
        self.add_score = 1
        if self.drawing:
            self.canvas.itemconfig(self.id, text=self.score)

    def hit(self):
        self.score += self.add_score
        self.add_score += 0
        if self.drawing:
            self.canvas.itemconfig(self.id, text=self.score)

    def hit_wall(self):
        self.score -= 1


class Game:
    def __init__(self, drawing=False):
        self.drawing = drawing
        if self.drawing:
            # Окно
            self.tk = Tk()
            # название окна
            self.tk.title('Game')
            self.tk.geometry("500x400+0+0")
            # Запретить менять размеры окна
            self.tk.resizable(0, 0)
            # Поверх других окон
            self.tk.wm_attributes('-topmost', 1)
            # Создаем новых холст где будем рисовать игру
            self.canvas = Canvas(self.tk, width=500, height=400, highlightthickness=0)
            self.canvas.pack()
            self.tk.update()
            # будут свои отдельные координаты
        else:
            self.canvas = {'width': 500, 'height': 400}
        self.first_score = 0
        self.last_score = 0
        # Говорим холсту, что у каждого видимого элемента
        # обновляем окно с холстом
        self.score = Score(self.canvas, 'green', drawing)
        self.paddle = Paddle(self.canvas, self.score, 'White', drawing)
        self.ball = Ball(self.canvas, self.paddle, self.score, 'red', drawing)
        self.ball_pos = [0, 0]
        self.paddle_pos = [0, 0]

    def reset(self):
        self.paddle.reset()
        self.score.reset()
        self.ball.reset()
        if self.drawing:
            self.tk.update_idletasks()
            self.tk.update()

        self.first_score = 0
        self.last_score = 0
        return self.get_state()

    def get_state(self):
        self.previous_ball_poss = self.ball_pos
        self.previous_paddle_pos = self.paddle_pos
        self.ball_pos = [(self.ball.pos[2] + self.ball.pos[0]) / 2 / 500,
                    (self.ball.pos[3] + self.ball.pos[1]) / 2 / 400]
        self.paddle_pos = [(self.ball.pos[2] + self.ball.pos[0]) / 2 / 500,
                      (self.ball.pos[3] + self.ball.pos[1]) / 2 / 400]

        return np.array(self.ball_pos + self.previous_ball_poss + self.paddle_pos + self.previous_paddle_pos)

    def stop(self):
        if self.drawing:
            self.tk.destroy()

    def step(self, action):
        if not self.ball.hit_bottom:
            self.paddle.turn(action)
            self.ball.draw()
            self.paddle.draw()
            if self.drawing:
                self.tk.update_idletasks()
                self.tk.update()
                time.sleep(0.05)
            self.first_score = self.last_score
            self.last_score = self.score.score
            delta_score = self.last_score - self.first_score
            if self.last_score >= 500:
                # self.stop()
                return self.get_state(), 100, True, 0
            return self.get_state(), delta_score, False, 0
        else:
            # self.stop()
            return self.get_state(), -1, True, 0
