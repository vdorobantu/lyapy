"""Base class for outputs with proportional and derivative components."""

from .output import Output

class PDOutput(Output):
    """Base class for outputs with proportional and derivative components.

    Override eta, proportional, derivative.

    Let n be the number of states, k be the proportional/derivative error sizes.
    """

    def proportional(self, x, t):
        """Compute proportional error component of output dynamics.

        Outputs a numpy array (k,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass

    def derivative(self, x, t):
        """Compute derivative error component of output dynamics.

        Outputs a numpy array (k,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass
