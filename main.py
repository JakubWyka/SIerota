from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet
import pygame
from pygame.locals import *

#testing code
if __name__ == "__main__":
    pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 10)
    state = pong.state
    while not pong.done:
        state_new, reward, done = pong.execute(0)
        pong.draw()
        state = state_new
        pong.update_clock()
    pong.exit_game()