"""Class for Control Lyapunov Functions (CLFs) of the form V(eta) = eta' P eta."""

from numpy import dot, identity
from numpy.linalg import eigvals
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

from .control_lyapunov_function import ControlLyapunovFunction
from .quadratic_lyapunov_function import QuadraticLyapunovFunction

class QuadraticControlLyapunovFunction(QuadraticLyapunovFunction, ControlLyapunovFunction):
    """Class for Control Lyapunov Functions (CLFs) of the form V(eta) = eta' P eta.

    Let n be the number of states, m be the number of inputs, p be the output
    vector size.

    Attributes:
    Control task output, output: AffineDynamicOutput
    Positive definite matrix, P: numpy array (p, p)
    Convergence rate, alpha: float
    """

    def __init__(self, affine_dynamic_output, P, alpha):
        """Initialize a QuadraticControlLyapunovFunction.

        Inputs:
        Control task output, affine_dynamic_output: AffineDynamicOutput
        Positive definite matrix, P: numpy array (p, p)
        Convergence rate, alpha: float
        """

        QuadraticLyapunovFunction.__init__(self, affine_dynamic_output, P)
        ControlLyapunovFunction.__init__(self, affine_dynamic_output)
        self.alpha = alpha

    def drift(self, x, t):
        """Evaluate the Lyapunov function drift for a state and time.

        Lyapunov function drift is grad_V(x, t) * output.drift(x, t).

        Outputs a float.

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        return dot(self.grad_V(x, t), self.output.drift(x, t))

    def decoupling(self, x, t):
        """Evaluate the Lyapunov function drift for a state and time.

        Lyapunov function drift is grad_V(x, t) * output.decoupling(x, t).

        Outputs a numpy array (m,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        return dot(self.grad_V(x, t), self.output.decoupling(x, t))

    def V_dot(self, x, u, t):
        return self.drift(x, t) + dot(self.decoupling(x, t), u)

    def build_ctle(feedback_linearizable_output, K, Q):
        """Build a quadratic CLF from a FeedbackLinearizableOutput with auxilliary control gain matrix, by solving the continuous time Lyapunov equation (CTLE).

        CTLE is

        A_cl' P + P A_cl = -Q

        for specified Q.

        Outputs a QuadraticControlLyapunovFunction.

        Inputs:
        Auxilliary control gain matrix, K: numpy array (k, p)
        Positive definite matrix for CTLE, Q: numpy array (p, p)
        """

        A = feedback_linearizable_output.closed_loop_dynamics(K)
        P = solve_continuous_lyapunov(A.T, -Q)
        alpha = min(eigvals(Q)) / max(eigvals(P))
        return QuadraticControlLyapunovFunction(feedback_linearizable_output, P, alpha)

    def build_care(feedback_linearizable_output, Q):
        """Build a quadratic CLF from a FeedbackLinearizableOutput with auxilliary control gain matrix, by solving the continuous algebraic Riccati equation (CARE).

        CARE is

        F'P + PF - PGG'P = -Q

        for specified Q.

        Outputs a QuadraticControlLyapunovFunction.

        Inputs:
        Positive definite matrix for CTLE, Q: numpy array (p, p)
        """

        F = feedback_linearizable_output.F
        G = feedback_linearizable_output.G
        R = identity(G.shape[1])
        P = solve_continuous_are(F, G, Q, R)
        alpha = min(eigvals(Q)) / max(eigvals(P))
        return QuadraticControlLyapunovFunction(feedback_linearizable_output, P, alpha)
