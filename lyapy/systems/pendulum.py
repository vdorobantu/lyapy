from numpy import array, sin

from .system import System

class Pendulum(System):
    def __init__(self, m, g, l):
        self.m, self.g, self.l = m, g, l

    def drift(self, x):
        theta, theta_dot = x
        return array([theta_dot, self.g / self.l * sin(theta)])

    def act(self, x):
        return array([[0], [1 / (self.m * (self.l ** 2))]])
