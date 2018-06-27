"""Inverted pendulum system.

System is modeled as a point mass at the end of a massless rod. Torque input is
applied at the opposite end. State x = [theta, theta_dot]', where theta is angle
of pendulum in rad clockwise from upright and theta_dot is the angular rate in
rad / sec. Input u = [tau]', where tau is torque in N * m. Equations of motion
are
    m * l ^ 2 * theta_ddot = m * g * l * sin(theta) + tau
where theta_ddot is the angular acceleration in rad / (sec ^ 2).
"""

from numpy import array, sin

from .system import System

class Pendulum(System):
    """Inverted pendulum system."""

    def __init__(self, m, g, l):
        """Initialize a Pendulum object.

        Inputs:
        Mass in kg, m: float
        Acceleration due to gravity in m / (sec ^ 2), g: float
        Length of pendulum in m, l: float
        """

        self.m, self.g, self.l = m, g, l

    def drift(self, x):
        theta, theta_dot = x
        return array([theta_dot, self.g / self.l * sin(theta)])

    def act(self, x):
        return array([[0], [1 / (self.m * (self.l ** 2))]])
