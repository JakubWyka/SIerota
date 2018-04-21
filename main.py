from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet
import pygame
from pygame.locals import *

from time import time
#MODE =0 granie z nauczenia sie,
#testing code
if __name__ == "__main__":
    EPISODES = 30
    agent=AI(Pong.OUTPUT_SHAPE[1], 1)
    batch_size = 50
    MODE = 0
    if MODE == 0:
        agent.load("./save.h5")
        for e in range(EPISODES):
            pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 5)
            if(MODE == 0):
                state = pong.state
                #end = time() + 1 * 60
                #while not pong.done and time() < end:
                while not pong.done:
                    act = agent.getAction(state)
                    state, my_action = pong.execute_ai(act)
                    #if reward == Pong.PONG_REWARD:
                      #  print(reward)
                    pong.draw()
                    #state = state_new
                 #   agent.remember(state, my_action)
                    pong.update_clock()
                #if len(agent.memory) > batch_size:
                 #   agent.replay(batch_size)
                  #  print(agent.epsilon)
                #if pong.end==True or time() >= end_learning:
                if pong.end == True:
                    break
            pong.exit_game()
    else:
        for e in range(EPISODES):
            pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 5)
            state = pong.state
            #end = time() + 1 * 60
            #while not pong.done and time() < end:
            while not pong.done:
                #act = agent.getAction(state)
                state, my_action = pong.execute()
                #if reward == Pong.PONG_REWARD:
                #  print(reward)
                pong.draw()
                #state = state_new
                agent.remember(state, my_action)
                pong.update_clock()
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                print(agent.epsilon)
            #if pong.end==True or time() >= end_learning:
            if pong.end == True:
                agent.save("./save.h5")
                pong.exit_game()
                break

