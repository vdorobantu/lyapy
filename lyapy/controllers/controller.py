"""Base class for controllers.

Let n be number of states, m be number of inputs.

Override u, V, dVdx, dV.
"""

class Controller:
    """Base class for controllers."""

    def u(self, x, t):
        """Evaluate the controller.

        Outputs a numpy array (m,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass

    def V(self, x, t):
        """Evaluate the Lyapunov function.

        Outputs a float.

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass

    def dVdx(self, x, t):
        """Evaluate the Lyapunov function gradient.

        Outputs a numpy array (n,).

        Inputs:
        State, x: numpy array (n,)
        Time, t: float
        """

        pass

    def dV(self, x, u, t):
        """Evaluate the Lyapunov function time derivative.

        Outputs a float.

        Inputs:
        State, x: numpy array (n,)
        Input, u: numpy array (m,)
        Time, t: float
        """

        pass
