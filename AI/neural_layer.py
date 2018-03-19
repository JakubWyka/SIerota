from .activation import ReLu
import numpy as np

class NeuralLayer():
    def __init__(self, inputs, outputs, prev=None):
        np.random.seed(1)
        self._weights = 2* np.random.random((inputs, outputs)) - 1
        if prev is not None:
            prev._next = self
        self._memory = None
        self._prev = prev
        self._next = None
        self._activation = ReLu(a=0.01) #LeakyReLu

    def feed_forward(self, input):
        output = self._activation(np.dot(input, self._weights))
        self._memory = (input, output)
        if self._next is not None:
            output = self._next.feed_forward(output)
        return output

    def backpropagate(self, error):
        input, output = self._memory
        delta = error * self._activation.derivative(output)       
        if self._prev is not None:
            error_prev = np.dot(delta, self._weights.T)
            self._prev.backpropagate(error_prev)
        self._weights += np.dot(input.T, delta)
