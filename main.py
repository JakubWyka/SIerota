from Games.pong import Pong
from AI.ai import AI
import pygame
from pygame.locals import *

from time import time

if __name__ == "__main__":
    EPISODES = 10
    agent=AI(Pong.OUTPUT_SHAPE[1], 2)
    batch_size = 2000
    agent.load('./save.anf')
    end_learning = time() + 3*60*60
    #agent.plot()
    odbil = 0
    wpuscil = 0
    for e in range(EPISODES):
        pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 3)
        state = pong.state
        bot_state = pong.bot_state
        end = time() + 1 * 60
        i = 0
        while not pong.done and time() < end:
            if i == 0:
                act = agent.getAction(state)
            state_new, bot_state_new, my_action, reward = pong.execute(act)
            if reward == Pong.ENEMY_SCORE_REWARD:
                wpuscil += 1
            elif reward == Pong.PONG_REWARD:
                odbil += 1
            pong.draw()
            if i == 0:
                agent.remember(bot_state, my_action)
            i = (i + 1) % 5
            bot_state = bot_state_new
            state = state_new
            pong.update_clock()
        #if len(agent.memory) > batch_size:
        #    agent.dumpTrainingData(batch_size)
        #    break
        if pong.end==True or time() >= end_learning:
            break
    #agent.save("./save.anf")
    print("wpuscil: {}, odbil: {}".format(wpuscil, odbil))
    pong.exit_game()

