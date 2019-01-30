"""Class for Lyapunov functions of the form V(eta) = eta' P eta."""

from numpy import dot

from .lyapunov_function import LyapunovFunction

class QuadraticLyapunovFunction(LyapunovFunction):
    """Class for Lyapunov functions of the form V(eta) = eta' P eta.

    Let n be the number of states, p be the output vector size.

    Attributes:
    Control task output, output: Output
    Positive definite matrix, P: numpy array (p, p)
    """

    def __init__(self, output, P):
        """Initialize a QuadraticLyapunovFunction object.

        Inputs:
        Control task output, output: Output
        Positive definite matrix, P: numpy array (p, p)
        """

        LyapunovFunction.__init__(self, output)
        self.P = P

    def V(self, x, t):
        eta = self.output.eta(x, t)
        return dot(eta, dot(self.P, eta))

    def grad_V(self, x, t):
        eta = self.output.eta(x, t)
        return 2 * dot(self.P, eta)
