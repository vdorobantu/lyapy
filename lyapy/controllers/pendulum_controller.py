"""Minimum norm Control Lypaunov Function (CLF) controller for inverted pendulum system.

The CLF is
    V = x' * P * x
where P is the solution to the Continuous Time Lypaunov Equation (CTLE) under
the closed loop dynamics for theta and theta_dot with linearizing feedback
control. The controller is
    u(x) = argmin_u (u ^ 2) / 2 s.t. V_dot(x, u) < -1 / lambda_1(P) * V(x)
where V_dot(x, u) is the time derivative of the CLF and lamda_1 is the maximum
eigenvalue of P.
"""

from numpy import array, dot, identity
from numpy.linalg import eigvals
from scipy.linalg import solve_continuous_lyapunov

from .controller import Controller
from ..systems import Pendulum

class PendulumController(Controller):
    """Minimum norm Control Lypaunov Function (CLF) controller for inverted pendulum system."""

    def __init__(self, pendulum, m_hat, K):
        """Initialize a PendulumController object.

        Inputs:
        Pendulum object, pendulum: Pendulum
        Estimate of pendulum mass in kg, m_hat: float
        Proportional and derivative contoller coefficients, K: numpy array (2,)
        """

        self.pendulum = Pendulum(m_hat, pendulum.g, pendulum.l)
        self.K = K
        A = array([[0, 1], -self.K])
        Q = identity(2)
        self.P = solve_continuous_lyapunov(A.T, -Q)

    def LfV(self, x):
        """Compute Lie derivative, L_fV(x) = (dV / dx) * f(x).

        Outputs a float.

        Inputs:
        State, x: numpy array (2,)
        """
        
        return 2 * dot(x, dot(self.P, self.pendulum.drift(x)))


    def LgV(self, x):
        """Compute Lie derivative, L_gV(x) = (dV / dx) * g(x)

        Outputs a float.

        Inputs:
        State, x: numpy array (2,)
        """

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
