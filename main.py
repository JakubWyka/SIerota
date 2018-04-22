from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet
import pygame
from pygame.locals import *

from time import time

#testing code
if __name__ == "__main__":
    EPISODES = 100000
    agent=AI(Pong.OUTPUT_SHAPE[1],2)
    batch_size = 50
    #agent.load("./q6h.h5")

    end_learning = time() + 6*60*60

    for e in range(EPISODES):
        pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 3)
        state = pong.state
        end = time() + 1 * 60
        while not pong.done and time() < end:
            act=agent.getAction(state)
            state_new, reward, done = pong.execute(act) 
            if reward == Pong.PONG_REWARD:
                print(reward)
            pong.draw()
            agent.remember(state, act, reward, state_new, done)
            state = state_new
            pong.update_clock()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            print(agent.epsilon)
        if pong.end==True or time() >= end_learning:
            break
    #agent.save("./save.h5")
    pong.exit_game()
