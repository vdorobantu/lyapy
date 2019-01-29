"""Base class for controllers."""

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

    def u(self, x, t):
        """Compute a control action.

        Outputs a numpy array (m,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """
        pass
