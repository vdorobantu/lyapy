"""Base class for Control Lyapunov Functions (CLFs)."""

from .lyapunov_function import LyapunovFunction

class ControlLyapunovFunction(LyapunovFunction):
    """Base class for Control Lyapunov Functions (CLFs).

    Override V, grad_V, V_dot.

    Attributes:
    Control task output, output: Output
    """

    def __init__(self, output):
        """Initialize a ControlLyapunovFunction object.

        Attributes:
        Control task output, output: Output
        """

        LyapunovFunction.__init__(self, output)

    def V_dot(self, x, u, t):
        """Evaluate Lyapunov function time derivative for a state, control input, and time.

        Returns a float.

        Inputs:
        State, x: numpy array (n,)
        Control input, u: numpy array (m,)
        Time, t: float
        """

        pass
