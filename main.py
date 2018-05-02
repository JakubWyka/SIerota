from Games.pong import Pong
from AI.ai import AI
import pygame
from pygame.locals import *

from time import time

#testing code
if __name__ == "__main__":
    EPISODES = 100000
    agent=AI(Pong.OUTPUT_SHAPE[1],2)
    batch_size = 64
    agent.load("./save.h5")

    end_learning = time() + 2*60*60

    for e in range(EPISODES):
        pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 3)
        state = pong.state
        end = time() + 1 * 60
        reward_per_act = 0
        i = 0
        while not pong.done and time() < end:
            if i == 0:
                act=agent.getAction(state)
            state_new, reward, done = pong.execute(act) 
            pong.draw()
            reward_per_act += reward
            if i == 4:
                agent.remember(state, act, reward_per_act, state_new, done)
                state = state_new
                reward_per_act = 0
            i = (i+1) % 5
            pong.update_clock()
        #if len(agent.memory) > batch_size:
        #    agent.replay(batch_size)
        if pong.end==True or time() >= end_learning:
            break
    #agent.plot()
    #agent.save("./save.h5")
    pong.exit_game()
