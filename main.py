from Games.pong import Pong
from AI.ai import AI
from AI.neuralnet import NeuralNet

#testing code
if __name__ == "__main__":
    import numpy as np
    i = np.array([[1,2,1], [2,1,1], [3,1,2], [1,1,1]])
    o = np.array([1,2,3,1])
    a = NeuralNet([3, 5, 4,1])
    for t in range(100000):
        for n in range(4):
            q = a.predict(np.reshape(i[n], (1,3)))
            error = o[n] - q[0]
            a.train(error)
    for n in range(4):
        print(a.predict(np.reshape(i[n], (1,3))))
    print(a.predict(np.reshape([4,1,1], (1,3))))
    print(a.predict(np.reshape([1,3,1], (1,3))))
    