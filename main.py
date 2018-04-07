from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet
import pygame
from pygame.locals import *

import keras as k
import numpy


#testing code
if __name__ == "__main__":
    EPISODES =100
    pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 1)
    agent=AI(Pong.OUTPUT_SHAPE[1],2)

    for e in range(EPISODES):
        pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 1)
        state = pong.state
        batch_size = 50
        while not pong.done:
            act=agent.getAction(state)
            state_new, reward, done = pong.execute(act) 
            if reward > 2:
                print(reward)
            pong.draw()
            state = state_new
            agent.remember(state, act, reward, state_new, done)
            pong.update_clock()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            print(agent.epsilon)
        if pong.end==True:
            break
    agent.save("./save.h5")
    pong.exit_game()
