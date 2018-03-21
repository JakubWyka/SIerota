import pygame
import sys, os
from abc import ABC, abstractclassmethod

class Game(ABC):
    @abstractclassmethod
    def __init__(self, key_bindings, width, height, title):
        self._done = False
        self._key_bindings = key_bindings
        if sys.platform == 'win32' or sys.platform == 'win64':
            os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.display.init()
        pygame.font.init()
        pygame.mixer.init(buffer=0)

        self._screen_size = [width, height]
        pygame.display.set_caption(title)
        self._surface = pygame.display.set_mode(self._screen_size)

    @abstractclassmethod
    def start(self):
        pass

    @property
    def done(self):
        return self._done 

    @property
    def surface(self):
        return self._surface

    @abstractclassmethod
    def draw(self):
        pass

    @abstractclassmethod
    def execute(self, action):
        pass
    
    def exit_game(self):
        pygame.quit()
        sys.exit()