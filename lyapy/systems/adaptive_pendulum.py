"""Inverted pendulum system for adaptive control.

System is modeled as a point mass at the end of a massless rod. Torque input is
applied at the opposite end. State x = [psi, psi_dot, theta_hat]', where psi is
angle of pendulum in rad clockwise from upright, psi_dot is the angular rate in
rad / sec, and theta_hat is estimate of reciprocal of pendulum mass. Input
u = [tau]', where tau is torque in N * m. Equations of motion are
    m * l ^ 2 * theta_ddot = m * g * l * sin(theta) + tau
where theta_ddot is the angular acceleration in rad / (sec ^ 2). Estimator
dynamics are
    theta_hat_dot = gamma * (2 * x' * P * [0, 1 / l ^ 2]')
where theta_hat_dot is the estimator rate and P is the solution to the
Continuous Time Lyapunov Equation (CTLE) under the closed loop dynamics for psi
and psi_dot with linearizing feedback control.
"""

from numpy import array, dot, identity, sin
from scipy.linalg import solve_continuous_lyapunov

from .system import System

class AdaptivePendulum(System):
    """Inverted pendulum system for adaptive control."""

    def __init__(self, m, g, l, K, gamma):
        """Initilize an AdaptivePendulum object.

        Inputs:
        Mass in kg, m: float
        Acceleration due to gravity in m / sec ^ 2, g: float
        Length of pendulum in m, l: float
        Proportional and derivative contoller coefficients, K: numpy array (2,)
        Esimator dynamics coefficient, gamma: float
        """
        
        self.m, self.g, self.l, self.K, self.gamma = m, g, l, K, gamma
        A = array([[0, 1], -K])
        Q = identity(2)
        self.P = solve_continuous_lyapunov(A.T, -Q)

    def drift(self, x):
        psi, psi_dot, _ = x
        return array([psi_dot, self.g / self.l * sin(psi), 0])

    def act(self, x):
        return array([[0], [1 / (self.m * (self.l ** 2))], [2 * self.gamma * dot(x[0:2], dot(self.P, array([0, 1 / (self.l ** 2)])))]])
