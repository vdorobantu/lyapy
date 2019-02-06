"""Base class for affine control systems of the form x_dot = f(x) + g(x) * u(x, t)."""

from numpy import dot

from .control_system import ControlSystem

class AffineControlSystem(ControlSystem):
    """Base class for affine control systems of the form x_dot = f(x) + g(x) * u.

    Let n be the number of states, m be number of control inputs.

    Override drift, act.
    """

    def __init__(self):
        ControlSystem.__init__(self)

    def f(self, x, u, t):
        return self.drift(x) + dot(self.act(x), u)

    def drift(self, x):
        """Compute drift function at a state.

        Outputs a numpy array (n,).

        Inputs:
        State, x: numpy array (n,)
        """

        pass

    def act(self, x):
        """Compute actuation matrix at a state.

        Outputs a numpy array (n, m).

        Inputs:
        State:, x: numpy array (n,)
        """

        pass
