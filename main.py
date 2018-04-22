from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet
import pygame
from pygame.locals import *

from time import time

if __name__ == "__main__":
    EPISODES = 100000
    agent=AI(Pong.OUTPUT_SHAPE[1], 2)
    batch_size = 50
    #agent.load(./s6h.h5)
    end_learning = time() + 6*60*60

    for e in range(EPISODES):
        pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 1)
        state = pong.state
        bot_state = pong.bot_state
        end = time() + 1 * 60
        while not pong.done and time() < end:
            act = agent.getAction(state)
            state_new, bot_state_new, my_action = pong.execute(act)
            pong.draw()
            agent.remember(bot_state, my_action)
            bot_state = bot_state_new
            state = state_new
            pong.update_clock()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if pong.end==True or time() >= end_learning:
            agent.save("./save.h5")
            pong.exit_game()
            break

