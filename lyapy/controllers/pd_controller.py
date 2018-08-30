"""Proportional derivative controller for any system.

"""

from numpy import array, dot, identity, reshape
from numpy import concatenate
from numpy.linalg import eigvals, norm
from scipy.linalg import solve_continuous_lyapunov

from .controller import Controller

class PdController(Controller):
    """PD controller for any system."""

    def __init__(self, K, r, r_dot):
        """Initialize a PdController object.

        Inputs:
        Proportional and derivative controller coefficients, K: numpy array (n, n/2)
        Angle trajectory, r: float -> numpy array (n/2,)
        Angular velocity trajectory, r_dot: float -> numpy array (n/2,)
        """
        self.K = K
        self.r, self.r_dot = r, r_dot

    def dVdx(self, x, t):
        pass

    def u(self, x, t):
        print(t)
        nstates = array([self.r(t)]).size
        e1 = array(x[0:nstates] - self.r(t))
        e2 = array(x[nstates:2*nstates] - self.r_dot(t))
        return dot(-self.K[0:nstates], e1) + dot(-self.K[nstates:2*nstates], e2)


    def V(self, x, t):
        pass

    def dV(self, x, u, t):
        pass
