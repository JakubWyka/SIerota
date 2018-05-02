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
        self.memory = deque(maxlen=86400)
        self.model = self._build_model()

    def _build_model(self):
        model = k.models.Sequential()
        model.add(k.layers.Dense(64, input_dim=self.state_size, kernel_initializer='normal', activation="relu"))
        model.add(k.layers.Dense(64, kernel_initializer='normal', activation="relu"))
        model.add(k.layers.Dense(64, kernel_initializer='normal', activation="relu"))
        model.add(k.layers.Dense(self.action_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer='adam')
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
        x = nump.zeros((batch_size, self.state_size))
        y = nump.zeros((batch_size, self.action_size))        
        i = 0
        for state, my_action in minibatch:
            target = [0,0]
            target[my_action] = 1
            x[i] = nump.reshape(state, (self.state_size,))
            y[i] = nump.reshape(target, (self.action_size,))
            i += 1
        history = self.model.fit(x, y, batch_size=64, shuffle=True, epochs=300, verbose=1)
        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.title('Uczenie z nauczycielem')
        plt.ylabel('błąd')
        plt.xlabel('iteracja')
        #plt.legend(['train'], loc='upper left')
        plt.savefig('supervisedlearning.png')

