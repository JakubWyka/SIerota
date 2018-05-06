from AI.Anfis.anfis import ANFIS, predict
from AI.Anfis.membership.MemFuncs import MemFuncs
import numpy as nump
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle
class AI():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10800)
        #self.model = self._build_model()

    def _build_model(self):
        mf = [[['gaussmf', {'mean': 0., 'sigma': 170.}], ['gaussmf', {'mean': 800., 'sigma': 170.}]],
                [['gaussmf', {'mean': 0., 'sigma': 150.}], ['gaussmf', {'mean': 600., 'sigma': 150.}]],
                [['gaussmf', {'mean': 0., 'sigma': 170.}], ['gaussmf', {'mean': 800., 'sigma': 170.}]],
                [['gaussmf', {'mean': 0., 'sigma': 70.}], ['gaussmf', {'mean': 300., 'sigma': 70.}], ['gaussmf', {'mean': 600., 'sigma': 70.}]],
                [['gaussmf', {'mean': 0., 'sigma': 70.}], ['gaussmf', {'mean': 300., 'sigma': 70.}], ['gaussmf', {'mean': 600., 'sigma': 70.}]]]

        X = nump.loadtxt('x.txt')
        Y = nump.loadtxt('target.txt')
        print(nump.shape(X))
        print(nump.shape(Y))

        mfc = MemFuncs(mf)
        anf = ANFIS(X, Y, mfc)
        anf.trainHybridJangOffLine(epochs=10)
        anf.plotErrors()
        return anf

    def remember(self, state, my_action):
        self.memory.append((state, my_action))

    def getAction(self, state):
        p = predict(self.model, state)
        r = 1 / (1 + nump.exp(-p[0][0])) #normalize output to [0,1]
        print(r)
        if p[0][0] >= 0.5:
            return 1
        else:
            return 0
    
    def load(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))

    def plot(self):
        self.model.plotErrors()
        x = nump.zeros((801,))
        for i in range(801):
            x[i] = i
        for i in range(self.state_size):
            self.model.plotMF(x, i)

    def dumpTrainingData(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        x = nump.zeros((batch_size, self.state_size))
        y = nump.zeros((batch_size, ))        
        i = 0
        for state, my_action in minibatch:
            x[i] = nump.reshape(state, (self.state_size,))
            y[i] = my_action
            i += 1
        nump.savetxt('x.txt', x)
        nump.savetxt('target.txt', y)


        


