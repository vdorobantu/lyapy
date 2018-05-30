from numpy import array, dot, identity, sin
from scipy.linalg import solve_continuous_lyapunov

from .controller import Controller
from ..systems import Pendulum

class PendulumController(Controller):
    def __init__(self, pendulum, m_hat, K):
        self.pendulum = Pendulum(m_hat, pendulum.g, pendulum.l)
        self.K = K
        A = array([[0, 1], -self.K])
        Q = identity(2)
        self.P = solve_continuous_lyapunov(A.T, -Q)

    def LfV(self, x):
        return 2 * dot(x, dot(self.P, self.pendulum.drift(x)))

    def LgV(self, x):
        return 2 * dot(x, dot(self.P, self.pendulum.act(x)))

    def synthesize(self):

        def u(x, t):
            theta = x[0]
            LgLf = 1 / (self.pendulum.m * (self.pendulum.l ** 2))
            Lf2 = self.pendulum.g / self.pendulum.l * sin(theta)
            return array([1 / LgLf * (-Lf2 - dot(self.K, x))])

        def V(x, t):
            return dot(x, dot(self.P, x))

        def dV(x, u, t):
            x_dot = self.pendulum.drift(x) + dot(self.pendulum.act(x), u)
            return 2 * dot(x, dot(self.P, x_dot))

        return u, V, dV
