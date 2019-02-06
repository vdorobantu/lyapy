"""Base class for Lyapunov Functions."""

class LyapunovFunction:
    """Base class for Lyapunov Functions.

    Override V, grad_V.

    Let n be the number of states, p be the output vector size.

    Attributes:
    Control task output, output: Output
    """

    def __init__(self, output):
        """Initialize a LyapunovFunction object.

        Inputs:
        Control task output, output: Output
        """

        self.output = output

    def V(self, x, t):
        """Evaluate Lyapunov function at a state and time.

        Outputs a float.

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass

    def grad_V(self, x, t):
        """Evaluate Lyapunov function spatial gradient at a state and time.

        Outputs a numpy array (p,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass
