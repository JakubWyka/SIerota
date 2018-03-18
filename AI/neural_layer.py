import numpy as np

class NerualLayer():
    def __init__(self, inputs, outputs, activation=None, prev=None):
        self._matrix = 2* np.random.random((inputs, outputs)) - 1
        if prev is not None:
            prev._next = self
        self._prev = prev
        self._next = None
        self._activation = activation
    
    def feed_forward(self, input):
        output = np.dot(input, self._matrix)
        if self._activation is not None:
            output = self._activation(output)
        if self._next is not None:
            output = self._next.feed_forward(output)
        return output
    
    def backpropagation(self):
        pass
