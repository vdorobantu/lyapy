"""Base class for control systems of the form x_dot = f(t, x, u)."""

from numpy import array, zeros

from .system import System

class ControlSystem(System):
    """Base class for control systems of the form x_dot = f(x, u, t).

    Override f.

    Let n be the number of states, m be number of control inputs.

    Attributes:
    Current control action, u: numpy array (m,)
    """

    def __init__(self):
        self.u = None

    def f(self, x, u, t):
        """Evaluate state derivative for a state, control action, and time.

        Outputs a numpy array (n,).

        Inputs:
        State, x: numpy array (n,)
        Control action, u: numpy array (m,)
        Time, t: float
        """

        pass

    def dx(self, t, x):
        return self.f(x, self.u, t)

    def simulate(self, x_0, controller, t_eval):
        """Simulate closed-loop system using Runge-Kutta 4,5.

        Solution is evaluated at N time steps. Outputs times and corresponding
        solutions as numpy array (N,) * numpy array (N, n).

        Inputs:
        Initial condition, x_0: numpy array (n,)
        Controller object, controller: Controller
        Solution times, t_eval: numpy array (N,)
        """

        T = len(t_eval) - 1
        n, m = len(x_0), len(controller.u(x_0, t_eval[0]))

        xs = zeros((T, n))
        us = zeros((T, m))
        ts = zeros((T,))

        for k, (t_0, t_1) in enumerate(zip(t_eval[:-1], t_eval[1:])):
            t_span = array([t_0, t_1])
            self.u = controller.u(x_0, t_0)

            xs[k] = x_0
            us[k] = self.u
            ts[k] = t_0

            _, xs_sim = super().simulate(x_0, t_span)
            x_0 = xs_sim[-1]

        return ts, xs
