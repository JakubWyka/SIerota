from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    def __call__(self, x, deriv = False):
        if deriv:
            return self.derivative(x)
        else:
            return self._function(x)

    @abstractmethod
    def derivative(self, x):
        pass
    
    @abstractmethod
    def _function(self, x):
        pass

class Linear(ActivationFunction):
    def __init__(self, a = 1):
        self._a = a

    def derivative(self, x):
        return self._a * np.ones_like(x)
    
    def _function(self, x):
        return self._a * x
        
class ReLu(ActivationFunction):
    def __init__(self, a = 0):
        self._a = a
    
    def derivative(self, x):
        x = 1. * (x > 0)
        x = self._a * (x != 1.)
        return x
    
    def _function(self, x):
        return np.maximum(x, self._a * x)

class Sigmoid(ActivationFunction):
    def derivative(self, x):
        return x*(1-x)
    def _function(self, x):
        return 1/(1+np.exp(-x))