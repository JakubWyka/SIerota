import pygame
from pygame.locals import *
from numpy import reshape
import sys, traceback
import random
import math
from .game import Game


def rndint(x):
    return int(round(x))


def clamp(x, minimum, maximum):
    if x < minimum: return minimum
    if x > maximum: return maximum
    return x


class Pong(Game):
    PADDLE_SPEED = 300 * 2
    BALL_SPEED = 200.0 * 2
    NO_REWARD = 0
    ENEMY_SCORE_REWARD = -1
    PONG_REWARD = 1 # given with time_reward
    SCORE_REWARD = 0
    CENTER_REWARD = 0.6
    OUTPUT_SHAPE = (1, 4)

    def __init__(self, key_bindings, max_score):
        super(Pong, self).__init__(key_bindings, 800, 600, "Pong - SI")
        self._font = pygame.font.SysFont("Times New Roman", 18)
        self._max_score = max_score
        self.start()

    def start(self):
        self._dt = 1.0 / 60.0
        self._done = False
        self.end = False      #to testing episodes
        self._ball = Pong.Ball(self._screen_size[0] / 2, self._screen_size[1] / 2, Pong.BALL_SPEED)
        self._player = Pong.Player((0, 255, 0),  Pong.Paddle(self._screen_size[0] - 5 - 10, self._screen_size[1] / 2 - 30, 10, 100, K_DOWN, K_UP))
        self._bot = Pong.Bot((0, 0, 255), Pong.Paddle(
            5, self._screen_size[1] / 2 - 30, 10, 450, K_s, K_w))
        self._clock = pygame.time.Clock()

    def update_clock(self):
        self._clock.tick(60)
        self._dt = 1.0 / clamp(self._clock.get_fps(), 30, 90)

    @property
    def state(self):
        x = self._ball.pos['x'] // (self._screen_size[0] // 4)
        y = self._ball.pos['y'] // (self._screen_size[1] // 4)
        bpos = y * 4 + x + 1
        state = [self._bot._paddle._pos['y'], bpos, self._ball.speed['x'], self._ball.speed['y']]
        return reshape(state, Pong.OUTPUT_SHAPE)

    def update(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self._done = True
                self.end=True
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self._done = True

        if self._player.score == self._max_score or self._bot.score == self._max_score:
            self._done = True

        restart = False
        given = False
        reward = Pong.NO_REWARD
        for _ in range(20):
            self._ball.update(self._dt / 20)
            if self._ball.pos['x'] < 0:
                self._bot.add_score()
                reward = Pong.SCORE_REWARD
                restart = True
            elif self._ball.pos['x'] > self._screen_size[0]:
                self._player.add_score()
                restart = True
                reward = Pong.ENEMY_SCORE_REWARD
                reward *= (self._player._paddle._pos['y'] - self._ball.pos['y'])/self._screen_size[1]

            if self._ball.pos['y'] < 0 or self._ball.pos['y'] > self._screen_size[1]:
                self._ball.pos['y'] = clamp(self._ball.pos['y'], 0, self._screen_size[1])
                self._ball.speed['y'] *= -1

            if restart:
                self._ball = Pong.Ball(self._screen_size[0] / 2, self._screen_size[1] / 2, Pong.BALL_SPEED)
            else:
                tmp = self._player.collide(self._ball)
                if tmp > Pong.NO_REWARD and not given:
                    given = True
                    reward = Pong.PONG_REWARD
                self._bot.collide(self._ball)
        return reward

    def draw(self):
        self.surface.fill((0, 0, 0))

        self._ball.draw()
        self._bot.draw()
        self._player.draw()

        p1_score_text = self._font.render("Score " + str(self._player.score), True, (255, 255, 255))
        p2_score_text = self._font.render("Score " + str(self._bot.score), True, (255, 255, 255))
        self._surface.blit(p1_score_text, (20, 20))
        self._surface.blit(p2_score_text, (self._screen_size[0] - p2_score_text.get_width() - 20, 20))

        pygame.display.flip()

    def execute(self, action):
        keys = pygame.key.get_pressed()
        self._player.update(self._dt, key=self._key_bindings[action])
        self._bot.update(self._dt, self._ball.pos['y'])
        self._ball.update(self._dt)
        reward = self.update()
        dy = abs(self._player._paddle._pos['y'] - self._screen_size[1]/2.0)
        reward += (1 - dy) * Pong.CENTER_REWARD
        return self.state, reward, self.done

    class Paddle:
        def __init__(self, x, y, w, h, key_d, key_u):
            self._pos = {'x': x, 'y': y}
            self._dim = {'width': w, 'height': h}
            self.key_d = key_d
            self.key_u = key_u

        def move(self, speed, dt):
            #oldy = self._pos['y']
            self._pos['y'] = clamp(self._pos['y'] - dt * speed, 0, Pong._screen_size[1] - self._dim['height'])
            #return self._pos['y'] - oldy

        def update_with_key(self, key, dt):
            if self.key_d == key:
                self.move(-1 * Pong.PADDLE_SPEED, dt)
            elif self.key_u == key:
                self.move(Pong.PADDLE_SPEED, dt)

        def update_with_keys(self, keys, dt):
            if keys[self.key_d]:
                self.move(-1 * Pong.PADDLE_SPEED, dt)
            elif keys[self.key_u]:
                self.move(Pong.PADDLE_SPEED, dt)

        def collide(self, ball):
            reward = Pong.NO_REWARD
            if ball.pos['x'] > self._pos['x'] and ball.pos['x'] < self._pos['x'] + self._dim['width'] and \
                            ball.pos['y'] > self._pos['y'] and ball.pos['y'] < self._pos['y'] + self._dim['height']:
                dist_lrdu = [
                    ball.pos['x'] - self._pos['x'],
                    (self._pos['x'] + self._dim['width']) - ball.pos['x'],
                    (self._pos['y'] + self._dim['height']) - ball.pos['y'],
                    ball.pos['y'] - self._pos['y'],
                ]
                reward = Pong.PONG_REWARD
                dist_min = min(dist_lrdu)
                if dist_min == dist_lrdu[0]:
                    ball.speed['x'] = -abs(ball.speed['x'])
                elif dist_min == dist_lrdu[1]:
                    ball.speed['x'] = abs(ball.speed['x'])
                elif dist_min == dist_lrdu[2]:
                    ball.speed['y'] = abs(ball.speed['y'])
                elif dist_min == dist_lrdu[3]:
                    ball.speed['y'] = -abs(ball.speed['y'])
            return reward

        def draw(self, color):
            pygame.draw.rect(Pong._surface, color,
                             (self._pos['x'], self._pos['y'], self._dim['width'], self._dim['height']), 0)
            pygame.draw.rect(Pong._surface, (255, 255, 255),
                             (self._pos['x'], self._pos['y'], self._dim['width'], self._dim['height']), 1)

    class Player:
        def __init__(self, color, paddle):
            self._score = 0
            self._color = color
            self._paddle = paddle

        def add_score(self):
            self._score += 1

        @property
        def score(self):
            return self._score

        def draw(self):
            self._paddle.draw(self._color)

        def update(self, dt, key=None, keys=None):
            if key is not None:
               self._paddle.update_with_key(key, dt)
            elif keys is not None:
                self._paddle.update_with_keys(keys, dt)

        def collide(self, ball):
            return self._paddle.collide(ball)

    class Bot(Player):
        COUNT = 0
        SPEED = 0

        def update(self, dt, poy):
            if self._paddle._pos['y']>poy:
                self.SPEED = 1.3
            else:
                self.SPEED = -1.3
            self._paddle.move(1.5 * self.SPEED * Pong.PADDLE_SPEED, dt)
            self.COUNT += 1

    class Ball:
        def __init__(self, x, y, speed):
            self.pos = {'x': x, 'y': y}

            angle = math.pi / 2
            while abs(math.cos(angle)) < 0.2 or abs(math.cos(angle)) > 0.8:
                angle = math.radians(random.randint(0, 360))
            self.speed = {'x': speed * math.cos(angle), 'y': speed * math.sin(angle)}

            self.radius = 4

        def update(self, dt):
            self.pos['x'] += dt * self.speed['x']
            self.pos['y'] += dt * self.speed['y']

        def speed_up(self):
            factor = 1.1
            self.speed['x'] *= factor
            self.speed['y'] *= factor

        def draw(self):
            pygame.draw.circle(Pong._surface, (255, 255, 255), [rndint(self.pos['x']), rndint(self.pos['y'])],
                               self.radius)
