#from .neuralnet import NeuralNet
import keras as k
import numpy as nump
import random
from collections import *
class AI():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()

    def _build_model(self):
        model = k.models.Sequential()
        model.add(k.layers.Dense(64, input_dim=self.state_size, kernel_initializer='normal', activation="relu"))
        model.add(k.layers.Dense(64, kernel_initializer='normal', activation="relu"))
        model.add(k.layers.Dense(64, kernel_initializer='normal', activation="relu"))
        model.add(k.layers.Dense(self.action_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer='rmsprop')
        return model

    def remember(self, state, my_action):
        self.memory.append((state, my_action))

    def getAction(self, state):
        val = self.model.predict(state)
        print(val[0])
        return nump.argmax(val[0]) 

    def load(self, name):
        self.model = k.models.load_model(name)
    def save(self, name):
        self.model.save(name)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, my_action in minibatch:
            target = [0,0]
            target[my_action] = 1
            target = nump.reshape(target, (1,2))
            self.model.fit(state, target, epochs=1, verbose=0)
