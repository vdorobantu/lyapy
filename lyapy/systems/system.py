"""Base class for dynamical systems of the form x_dot = f(t, x)."""

from scipy.integrate import solve_ivp

class System:
    """Base class for dynamical systems of the form x_dot = f(t, x).

    Override dx.

    Let n be number of states.
    """

    def dx(self, t, x):
        """Evaluate state derivative at a time and state.

        Outputs a numpy array (n,).

        Inputs:
        Time, t: float
        State, x: numpy array (n,)
        """

        pass

    def simulate(self, x_0, t_eval, rtol=1e-6, atol=1e-6):
        """Simulate closed-loop system using Runge-Kutta 4,5.

        Solution is evaluated at N time steps. Outputs times and corresponding
        solutions as numpy array (N,) * numpy array (N, n).

        Inputs:
        Initial condition, x_0: numpy array (n,)
        Solution times, t_eval: numpy array (N,)
        RK45 relative tolerance, rtol: float
        RK45 absolute tolerance, atol: float
        """

        t_span = [t_eval[0], t_eval[-1]]
        sol = solve_ivp(self.dx, t_span, x_0, t_eval=t_eval, rtol=rtol, atol=atol)
        return sol.t, sol.y.T
