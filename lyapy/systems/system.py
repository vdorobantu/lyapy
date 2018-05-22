from numpy import dot
from scipy.integrate import solve_ivp

class System:
    def drift(self, x):
        pass

    def act(self, x):
        pass

    def dynamics(self, u):

        def dx(t, x):
            return self.drift(x) + dot(self.act(x), u(x, t))

        return dx

    def simulate(self, u, x_0, t_eval):
        dx = self.dynamics(u)
        t_span = [0, t_eval[-1]]
        sol = solve_ivp(dx, t_span, x_0, t_eval=t_eval, rtol=1e-6, atol=1e-6)
        return sol.t, sol.y.T
