"""Proportional derivative controller for any system.
The controller is
    u(x, t) = -K_p * (x - r(t)) - K_d * (x_dot - r_dot(t))
where
    K_p is the proportional controller coefficient, and is a positive definite matrix
    K_d is the derivative controller coefficient, and is a positive definite matrix
    r is the desired trajectories for x to follow
    r_dot is the desired trajectories for x_dot to follow

"""

from numpy import array, dot, identity, reshape
from numpy import concatenate
from numpy.linalg import eigvals, norm
from scipy.linalg import solve_continuous_lyapunov

from .controller import Controller

class PDController(Controller):
    """PD controller for any system."""

    def __init__(self, K, r, r_dot):
        """Initialize a PDController object.

        Inputs:
        Proportional and derivative controller coefficients, K: numpy array (2n, n)
        Angle trajectory, r: float -> numpy array (n,)
        Angular velocity trajectory, r_dot: float -> numpy array (n,)
        """
        self.K = K
        self.r, self.r_dot = r, r_dot

    def dVdx(self, x, t):
        pass

    def u(self, x, t):
        print(t)
        n = array([self.r(t)]).size
        e1 = array(x[0:n] - self.r(t))
        e2 = array(x[n:2*n] - self.r_dot(t))
        return dot(-self.K[0:n], e1) + dot(-self.K[n:2*n], e2)


    def V(self, x, t):
        pass

    def dV(self, x, u, t):
        pass
