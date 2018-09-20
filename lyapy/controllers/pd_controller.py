"""Proportional derivative controller for any system.
The controller is
    u(x, t) = -K_p * (x - r(t)) - K_d * (x_dot - r_dot(t))
where
    K_p is the positive definite matrix of proportional controller gains
    K_d is the positive definite matrix of derivative controller gains
    r is the desired trajectories for x to follow
    r_dot is the desired trajectories for x_dot to follow

"""

from numpy import dot
from numpy import concatenate

from .controller import Controller

class PDController(Controller):
    """PD controller for any system."""

    def __init__(self, K, r, r_dot):
        """Initialize a PDController object.

        Inputs:
        Proportional and derivative controller coefficients, K: numpy array (m, n)
        Angle trajectory, r: float -> numpy array (n,)
        Angular velocity trajectory, r_dot: float -> numpy array (n,)
        """
        self.K = K
        self.r, self.r_dot = r, r_dot

    def dVdx(self, x, t):
        pass

    def u(self, x, t):
        e = x - concatenate([self.r(t), self.r_dot(t)])
        return -dot(self.K, e)

    def V(self, x, t):
        pass

    def dV(self, x, u, t):
        pass
