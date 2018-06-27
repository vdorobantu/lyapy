"""Base class for controllers.

Let n be number of states, m be number of inputs.

Override synthesize.
"""

class Controller:
    """Base class for controllers."""

    def synthesize(self):
        """Synthesize controller, Lyapunov function, and Lyapunov function time
        derivative.

        Controller maps state and time to input, maps numpy array (n,) * float
        to numpy array (m,). Lyapunov function maps state and time to scalar,
        maps numpy array (n,) * float to float. Lyapunov function time
        derivative maps state, input, and time to scalar, maps numpy array (n,)
        * numpy array (m,) * float to float.
        """

        pass
