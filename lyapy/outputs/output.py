"""Base class for outputs."""

class Output:
    """Base class for outputs. 

    Override eta.

    Let n be the number of states, p be the output vector size.
    """

    def eta(self, x, t):
        """Evaluate output vector at a state and time.

        Outputs a numpy array (p,)

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass
