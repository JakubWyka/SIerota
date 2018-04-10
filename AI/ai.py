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
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()



    def _build_model(self):
        #input_shape
        self.model = k.models.Sequential()
        self.model.add(k.layers.Dense(24,input_dim = self.state_size, activation="relu"))
        self.model.add(k.layers.Dense(24, activation="relu"))
        self.model.add(k.layers.Dense(self.action_size, activation="linear"))
        self.model.compile(loss="mean_squared_error",
                            optimizer=k.optimizers.Adam(lr=0.1))
        return self.model

    def remember(self, state, action, reward, state_new, isdone):
        self.memory.append ((state, action, reward, state_new, isdone))

    def getAction(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        val = self.model.predict(state)
        return nump.argmax(val[0])  

    def load(self, name):
        #self.model.load_weights(name)
        self.model = k.models.load_model(name)
        self.epsilon = self.epsilon_min
    def save(self, name):
        #self.model.save_weights(name)
        self.model.save(name)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, state_new, isdone in minibatch:
            if not isdone:
                predVal=nump.amax(self.model.predict(state_new)[0])
                target = (reward + self.gamma *predVal)
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay