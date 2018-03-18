from neural_layer import NerualLayer

class NerualNet():
    def __init__(self, shape): #shape is a list of sizes for neural layers
        self._first = NerualLayer(shape[0], shape[1])
        tmp = self._first
        for i in range(1, len(shape) - 1):
            self._last = NerualLayer(shape[i], shape[i+1], prev=tmp)
            tmp = self._last

    def feed_forward(self, input):
        return self._first.feed_forward(input)

    def backpropagate(self):
        pass
