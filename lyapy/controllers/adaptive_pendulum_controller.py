"""Minimum norm Control Lypaunov Function (CLF) adaptive controller for inverted pendulum system.

The CLF is
    V = [psi, psi_dot] * P * [psi, psi_dot]'
where P is the solution to the Continuous Time Lypaunov Equation (CTLE) under
the closed loop dynamics for psi and psi_dot with linearizing feedback
control. The controller is
    u(psi, psi_dot, theta_hat)
        = argmin_u 1 / 2 * (u ^ 2)
          s.t. V_dot(theta, theta_hat, u) < -1 / lambda_1 * V(theta, theta_hat)
where V_dot(x, u) is the time derivative of the CLF and lambda_1 is the maximum
eigenvalue of P. Estimator dynamics are
    theta_hat_dot = gamma * (2 * x' * P * [0, 1 / l ^ 2]')
where theta_hat_dot is the estimator rate.
"""

from numpy import array, dot, sin
from numpy.linalg import eigvals

from .controller import Controller
from ..systems import AdaptivePendulum

class AdaptivePendulumController(Controller):
    """Minimum norm Control Lypaunov Function (CLF) adaptive controller for inverted pendulum system."""

    def __init__(self, adaptive_pendulum):
        """Initializes an AdaptivePendulumController object.

        Inputs:
        Adaptive pendulum system, adaptive_pendulum: AdaptivePendulum
        """
        
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
