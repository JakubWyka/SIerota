from neural_layer import NerualLayer
from activation import Linear

class NerualNet():
    def __init__(self, shape): #shape is a list of sizes for neural layers
        self._first = NerualLayer(inputs=shape[0], outputs=shape[1])
        tmp = self._first
        for i in range(1, len(shape) - 1):
            self._last = NerualLayer(inputs=shape[i], outputs=shape[i+1], prev=tmp)
            tmp = self._last
        self._last._activation = Linear()

    def predict(self, input):
        return self._first.feed_forward(input)[0]

    def train(self, error):
        self._last.backpropagate(error)

#testing code
if __name__ == "__main__":
    import numpy as np
    i = np.array([[1,2,1], [2,1,1], [3,1,2], [1,1,1]])
    o = np.array([1,2,3,1])
    a = NerualNet([3, 5, 4,1])
    for t in range(100000):
        for n in range(4):
            q = a.predict(np.reshape(i[n], (1,3)))
            error = o[n] - q[0]
            a.train(error)
    for n in range(4):
        print(a.predict(np.reshape(i[n], (1,3))))
    print(a.predict(np.reshape([4,1,1], (1,3))))
    print(a.predict(np.reshape([1,3,1], (1,3))))
    