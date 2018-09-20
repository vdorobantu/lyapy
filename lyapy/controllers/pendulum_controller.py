"""Minimum norm Control Lypaunov Function (CLF) controller for inverted pendulum system.

The CLF is
    V = (x - x_d(t))' * P * (x - x_d(t))
where P is the solution to the Continuous Time Lypaunov Equation (CTLE) under
the closed loop dynamics for theta and theta_dot with linearizing feedback
control, and x_d is a desired trajectory. The controller is
    u(x) = argmin_u (u ^ 2) / 2 s.t. V_dot(x, u, t) < -1 / lambda_1(P) * V(x, t)
where V_dot(x, u, t) is the time derivative of the CLF and lamda_1 is the
maximum eigenvalue of P.
"""

from numpy import array, dot, identity
from numpy.linalg import eigvals, norm
from scipy.linalg import solve_continuous_lyapunov

from .controller import Controller
from ..systems import Pendulum

class PendulumController(Controller):
    """Minimum norm Control Lypaunov Function (CLF) controller for inverted pendulum system."""

    def __init__(self, pendulum, K, r, r_dot, r_ddot):
        """Initialize a PendulumController object.

        Inputs:
        Estimated pendulum system, pendulum: Pendulum
        Proportional and derivative contoller coefficients, K: numpy array (2,)
        Angle trajectory, r: float -> float
        Angular velocity trajectory, r_dot: float -> float
        Angular acceleration trajectory, r_ddot: float -> float
        """

        self.pendulum = pendulum
        A = array([[0, 1], -K])
        Q = identity(2)
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.lambda_1 = max(eigvals(self.P))
        self.r, self.r_dot, self.r_ddot = r, r_dot, r_ddot

    def dVdx(self, x, t):
        e = x - array([self.r(t), self.r_dot(t)])
        return 2 * dot(e, self.P)

    def LfV(self, x, t):
        """Compute Lie derivative, L_fV(x) = (dV / dx) * (f(x) - [r_dot(t), r_ddot(t)]).

        Outputs a float.

        Inputs:
        State, x: numpy array (2,)
        Time, t: float
        """

        return dot(self.dVdx(x, t), self.pendulum.drift(x) - array([self.r_dot(t), self.r_ddot(t)]))

    def LgV(self, x, t):
        """Compute Lie derivative, L_gV(x) = (dV / dx) * g(x)

        Outputs a float.

        Inputs:
        State, x: numpy array (2,)
        Time, t: float
        """

        return dot(self.dVdx(x, t), self.pendulum.act(x))

    def u(self, x, t):
        print(t)
        tol = 1e-6
        LgV = self.LgV(x, t)
        # Dual optimal solution
        if norm(LgV) < tol:
            lambda_star = 0
        else:
            lambda_star = (self.LfV(x, t) + self.V(x, t) / self.lambda_1) / (norm(LgV) ** 2)
            lambda_star = max(0, lambda_star)
        return -lambda_star * LgV

    def V(self, x, t):
        e = x - array([self.r(t), self.r_dot(t)])
        return dot(e, dot(self.P, e))

    def dV(self, x, u, t):
        return self.LfV(x, t) + dot(self.LgV(x, t), u)
