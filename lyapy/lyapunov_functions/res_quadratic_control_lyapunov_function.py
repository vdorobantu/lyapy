"""Class for Rapidly Exponentially Stabilizing Control Lyapunov Functions (RES-CLFs)."""

from numpy import dot, identity
from numpy.linalg import eigvals
from scipy.linalg import block_diag

from .quadratic_control_lyapunov_function import QuadraticControlLyapunovFunction

class RESQuadraticControlLyapunovFunction(QuadraticControlLyapunovFunction):
    """Class for Rapidly Exponentially Stabilizing Control Lyapunov Functions (RES-CLFs)."""

    def __init__(self, robotic_system_output, P, Q, epsilon):
        M = block_diag(identity(robotic_system_output.k) / epsilon, identity(robotic_system_output.k))
        P = dot(M, dot(P, M)) * epsilon
        Q = dot(M, dot(Q, M))
        alpha = min(eigvals(Q)) / max(eigvals(P))
        QuadraticControlLyapunovFunction.__init__(self, robotic_system_output, P, alpha)

    def build_ctle(robotic_system_output, K, Q, epsilon=1):
        lyapunov_function = QuadraticControlLyapunovFunction.build_ctle(robotic_system_output, K, Q)
        return RESQuadraticControlLyapunovFunction(robotic_system_output, lyapunov_function.P, Q, epsilon)

    def build_care(robotic_system_output, Q, epsilon=1):
        lyapunov_function = QuadraticControlLyapunovFunction.build_care(robotic_system_output, Q)
        return RESQuadraticControlLyapunovFunction(robotic_system_output, lyapunov_function.P, Q, epsilon)
