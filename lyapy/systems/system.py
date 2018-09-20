"""Base class for dynamical systems."""

from numpy import dot
from scipy.integrate import solve_ivp

class System:
    """Base class for dynamical systems.

    Let n be number of states, m be number of inputs. Dynamics are defined as
        x_dot = f(x) + g(x)u
    for state x in R^n, input u in R^m, drift dynamics f, and actuation matrix g.

    Override drift, act.
    """

    def drift(self, x):
        """Evalutate drift dynamics, f(x).

        Outputs numpy array (n,).

        Inputs:
        State, x: numpy array (n,)
        """

        pass

    def act(self, x):
        """Evaluate actuation matrix, g(x).

        Outputs numpy array (n, m).

        Inputs:
        State, x: numpy array (n,)
        """

        pass

    def dynamics(self, u):
        """Return dynamics function, x_dot.

        Outputs function mapping float * numpy array (n,) to numpy array (n,).

        Inputs:
        Controller, u: float * numpy array (n,) -> numpy array (m,)
        """

        def dx(t, x):
            return self.drift(x) + dot(self.act(x), u(x, t))

        return dx

    def simulate(self, u, x_0, t_eval):
        """Simulate closed-loop system using Runge-Kutta 4,5.

        Solution is evaluated at N time steps. Outputs times and corresponding
        solutions as numpy array (N,) * numpy array (N, n).

        Inputs:
        Controller, u: numpy array (n,) * float -> numpy array (m,)
        Initial condition, x_0: numpy array (n,)
        Solution times, t_eval: numpy array (N,)
        """

        dx = self.dynamics(u)
        t_span = [t_eval[0], t_eval[-1]]
        sol = solve_ivp(dx, t_span, x_0, t_eval=t_eval, rtol=1e-6, atol=1e-6)
        print('Number of evals', sol.nfev)
        return sol.t, sol.y.T
