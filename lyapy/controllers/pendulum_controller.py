from numpy import array, dot, identity
from numpy.linalg import eigvals
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
        lambda_1 = max(eigvals(self.P))
        tol = 1e-6

        def V(x, t):
            return dot(x, dot(self.P, x))

        def u(x, t):
            if (abs(self.LgV(x)[0]) < tol) or (self.LfV(x) <= -1 / lambda_1 * V(x, t)):
                return array([0])
            return array([1 / (self.LgV(x)[0]) * (-self.LfV(x) - 1 / lambda_1 * V(x, t))])

        def dV(x, u, t):
            x_dot = self.pendulum.drift(x) + dot(self.pendulum.act(x), u)
            return 2 * dot(x, dot(self.P, x_dot))

        return u, V, dV
