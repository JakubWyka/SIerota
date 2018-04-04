from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet
import pygame
from pygame.locals import *

import keras as k
import numpy
#testing code
if __name__ == "__main__":
    pong = Pong(key_bindings = {0 : K_DOWN, 1: K_UP}, max_score = 10)
    state = pong.state

    #SI MODEL PROTOTYPE
    #TODO MOVE TO SI CLASS
    model = k.models.Sequential()
    model.add(k.layers.Dense(24, input_dim=Pong.OUTPUT_SHAPE[1], activation="relu"))
    model.add(k.layers.Dense(24, activation="relu"))
    model.add(k.layers.Dense(2, activation="linear"))
    model.compile(loss="mean_squared_error",
                            optimizer=k.optimizers.Adam(lr=0.1))

    p = model.predict(state)
    print(p)
    while not pong.done:
        #TODO MOVE PREDICTION TO SI CLASS
        p = model.predict(state)
        print(p)
        state_new, reward, done = pong.execute(numpy.argmax(p[0])) #TODO CHANGE CALLING NUMPY.ARGMAX TO CALLING METHOD FROM SI CLASS
        pong.draw()
        state = state_new
        
        pong.update_clock()
    pong.exit_game()
