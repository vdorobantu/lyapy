"""Base class for dynamical systems."""

from numpy import array, dot, zeros
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

        # def dx(t, x):
        #     return self.drift(x) + dot(self.act(x), u(x, t))

        def dx(t, x):
            return self.drift(x) + dot(self.act(x), u)

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

        # dx = self.dynamics(u)
        # t_span = [t_eval[0], t_eval[-1]]
        # sol = solve_ivp(dx, t_span, x_0, t_eval=t_eval, rtol=1e-6, atol=1e-6)
        # print('Number of evals', sol.nfev)
        # return sol.t, sol.y.T

        T = len(t_eval) - 1
        n, m = len(x_0), len(u(x_0, t_eval[0]))

        xs = zeros((T, n))
        us = zeros((T, m))
        ts = zeros((T,))

        for t, (t_0, t_1) in enumerate(zip(t_eval[:-1], t_eval[1:])):
            t_span = array([t_0, t_1])
            u_0 = u(x_0, t_0)

            xs[t] = x_0
            us[t] = u_0
            ts[t] = t_0

            dx = self.dynamics(u_0)
            sol = solve_ivp(dx, t_span, x_0, t_eval=t_span, rtol=1e-6, atol=1e-6)

            x_0 = sol.y.T[-1]

        return ts, xs
