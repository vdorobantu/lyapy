from numpy import array, dot, sin
from numpy.linalg import eigvals

from .controller import Controller
from ..systems import AdaptivePendulum

class AdaptivePendulumController(Controller):
    def __init__(self, adaptive_pendulum):
        self.adaptive_pendulum = adaptive_pendulum

    def synthesize(self):
        def V(x, t):
            return dot(x[0:2], dot(self.adaptive_pendulum.P, x[0:2]))

        def u(x, t):
            psi, theta_hat = x[0], x[-1]
            Lf2h = self.adaptive_pendulum.g / self.adaptive_pendulum.l * sin(psi)
            LgLfh = theta_hat / (self.adaptive_pendulum.l ** 2)
            return array([1 / LgLfh * (-Lf2h - dot(self.adaptive_pendulum.K, x[0:2]))])

        def dV(x, u, t):
            x_dot = self.adaptive_pendulum.drift(x) + dot(self.adaptive_pendulum.act(x), u)
            return 2 * dot(x, dot(self.adaptive_pendulum.P, x_dot[0:2]))

        return u, V, dV
