from abc import ABC, abstractclassmethod

class Game(ABC):
    @abstractclassmethod
    def __init__(self, duration):
        self._done = False
        self._duration = duration

    @abstractclassmethod
    def start(self):
        pass

    @property
    def done(self):
        return self._done 

    @abstractclassmethod
    def draw(self):
        pass

    @abstractclassmethod
    def execute(self, action):
        pass