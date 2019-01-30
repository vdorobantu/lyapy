"""Base class for outputs with dynamics that decompose affinely in the input."""

from .output import Output

class AffineDynamicOutput(Output):
    """Base class for outputs with dynamics that decompose affinely in the input. 

    Override eta, drift, decoupling.

    Output dynamics are eta_dot(x, t) = drift(x, t) + decoupling(x, t) * u

    Let n be the number of states, p be the output vector size.
    """

    def drift(self, x, t):
        """Evaluate drift function at a state and time.

        Outputs a numpy array (n,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass

    def decoupling(self, x, t):
        """Evaluate decoupling matrix at a state and time.

        Outputs a numpy array (n, m).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass
