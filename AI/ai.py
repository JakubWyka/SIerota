#from .neuralnet import NeuralNet
import keras as k
import numpy as nump
import random
from collections import *
import matplotlib.pyplot as plt
class AI():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3600)
        self.gamma = 0.80    # discount rate
        self.epsilon = 0.75  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.history = []



    def _build_model(self):
        #input_shape
        model = k.models.Sequential()
        model.add(k.layers.Dense(64,input_dim = self.state_size, activation="relu"))
        model.add(k.layers.Dense(64, activation="relu"))
        model.add(k.layers.Dense(64, activation="relu"))
        model.add(k.layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mean_squared_error",
                      optimizer=k.optimizers.Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, state_new, isdone):
        self.memory.append ((state, action, reward, state_new, isdone))

    def getAction(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        val = self.model.predict(state)
        print(val[0])
        return nump.argmax(val[0])  

    def load(self, name):
        self.model = k.models.load_model(name)
        self.epsilon = self.epsilon_min
    def save(self, name):
        self.model.save(name)

    def plot(self):
        plt.plot(self.history)
        plt.title('Q-learning')
        plt.ylabel('błąd')
        plt.xlabel('iteracja')
        #plt.legend(['train'], loc='upper left')
        plt.savefig('unsupervisedlearning.png')
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        x = nump.zeros((batch_size, self.state_size))
        y = nump.zeros((batch_size, self.action_size))
        i = 0
        for state, action, reward, state_new, isdone in minibatch:
            if not isdone:
                predVal=nump.amax(self.model.predict(state_new)[0])
                target = (reward + self.gamma *predVal)
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            y[i] = target_f
            x[i] = state
            i += 1
        h = self.model.fit(x, y, batch_size=32, shuffle=True, epochs=10, verbose=1)
        self.history.extend(h.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
