from numpy import array, dot, identity, sin
from scipy.linalg import solve_continuous_lyapunov

from .system import System

class AdaptivePendulum(System):
    def __init__(self, m, g, l, K, gamma):
        self.m, self.g, self.l, self.K, self.gamma = m, g, l, K, gamma
        A = array([[0, 1], -K])
        Q = identity(2)
        self.P = solve_continuous_lyapunov(A.T, -Q)

    def drift(self, x):
        psi, psi_dot, _ = x
        return array([psi_dot, self.g / self.l * sin(psi), 0])

    def act(self, x):
        return array([[0], [1 / (self.m * (self.l ** 2))], [2 * self.gamma * dot(x[0:2], dot(self.P, array([0, 1 / (self.l ** 2)])))]])
