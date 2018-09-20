from numpy import array, dot, identity
from numpy.linalg import eigvals, norm
from scipy.linalg import solve_continuous_lyapunov

from .controller import Controller

class SegwayController(Controller):
    def __init__(self, segway, K, r, r_dot, r_ddot):
        self.segway, self.r, self.r_dot, self.r_ddot = segway, r, r_dot, r_ddot
        A = array([[0, 1], -K])
        Q = identity(2)
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.tol = 1e-6
        self.lambda_1 = max(eigvals(self.P))

    def e(self, x, t):
        _, theta, _, theta_dot = x
        return array([theta - self.r(t), theta_dot - self.r_dot(t)])

    def dVdx(self, x, t):
        dVdeta = 2 * dot(self.e(x, t), self.P)
        return array([0, dVdeta[0], 0, dVdeta[1]])

    def LfV(self, x, t):
        return dot(self.dVdx(x, t), self.segway.drift(x) - array([0, self.r_dot(t), 0, self.r_ddot(t)]))

    def LgV(self, x, t):
        return dot(self.dVdx(x, t), self.segway.act(x))

    def V(self, x, t):
        e = self.e(x, t)
        return dot(e, dot(self.P, e))

    def u(self, x, t):
        LgV = self.LgV(x, t)
        if norm(LgV) < self.tol:
            return array([0])
        return (-self.LfV(x, t) - (1 / self.lambda_1) * self.V(x, t)) * LgV / (norm(LgV) ** 2)

    def dV(self, x, u, t):
        return self.LfV(x, t) + dot(self.LgV(x, t), u)
