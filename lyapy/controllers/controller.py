"""Base class for controllers."""

from numpy import array

class Controller:
    """Base class for controllers.

    Override u.

    Let n be the number of states, m be the number of inputs.

    Attributes:
    Control task output, output: Output
    """

    def __init__(self, output):
        """Initialize a Controller object.

        Inputs:
        Control task output, output: Output
        """
        self.output = output

    def u(self, x, t, update=True):
        """Compute a control action.

        Outputs a numpy array (m,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        Internal state update flag, update: bool
        """

        pass

    def reset(self):
        """Reset internal state of controller."""

        pass

    def evaluate(self, xs, ts):
        """Evaluate controller at given states and times.

        Let n be the number of states, m be the number of inputs, T be the
        number of states and times.

        Outputs a numpy array (T, m).

        Inputs:
        States, xs: numpy array (T, n)
        Times, ts: numpy array (T,)
        """

        self.reset()
        us = array([self.u(x, t) for x, t in zip(xs, ts)])

        return us
