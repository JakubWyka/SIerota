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
    def derivative(self, x):
        return np.ones_like(x)
    def _function(self, x):
        return x
        
class ReLu(ActivationFunction):
    def derivative(self, x):
        return 1. * (x > 0)
    def _function(self, x):
        return np.maximum(x, 0)

class Sigmoid(ActivationFunction):
    def derivative(self, x):
        return x*(1-x)
    def _function(self, x):
        return 1/(1+np.exp(-x))