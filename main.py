from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet
import pygame
from pygame.locals import *

from time import time

if __name__ == "__main__":
    EPISODES = 100000
    agent=AI(Pong.OUTPUT_SHAPE[1], 2)
    batch_size = 20000
    #agent.load('./s2.h5')
    end_learning = time() + 30*60

    for e in range(EPISODES):
        pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 3)
        state = pong.state
        bot_state = pong.bot_state
        end = time() + 1 * 60
        i = 0
        while not pong.done and time() < end:
            if i == 0:
                act = agent.getAction(state)
            state_new, bot_state_new, my_action = pong.execute(act)
            pong.draw()
            if i == 0:
                agent.remember(bot_state, my_action)
            i = (i + 1) % 5
            bot_state = bot_state_new
            state = state_new
            pong.update_clock()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            break
        if pong.end==True or time() >= end_learning:
            break
    agent.save("./save.h5")
    pong.exit_game()

