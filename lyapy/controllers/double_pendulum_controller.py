"""Minimum norm Control Lypaunov Function (CLF) controller for double inverted pendulum system.

The CLF is
    V = (x - x_d(t))' * P * (x - x_d(t))
where P is the solution to the Continuous Time Lypaunov Equation (CTLE) under
the closed loop dynamics for theta1, theta2, theta1_dot, and theta2_dot with linearizing feedback
control, and x_d is a desired trajectory. The controller is
    u(x) = argmin_u norm(u) ^ 2 s.t. V_dot(x, u, t) < -1 / lambda_1(P) * V(x, t)
where V_dot(x, u, t) is the time derivative of the CLF and lambda_1 is the
maximum eigenvalue of P.
"""

from numpy import array, block, concatenate, dot, identity, zeros
from numpy.linalg import eigvals, norm
from scipy.linalg import solve_continuous_lyapunov

from .controller import Controller
from ..systems import DoublePendulum

class DoublePendulumController(Controller):
    """Minimum norm Control Lypaunov Function (CLF) controller for double inverted pendulum system."""

    def __init__(self, double_pendulum, m_1_hat, m_2_hat, K_p, K_d, r, r_dot, r_ddot):
        """Initialize a DoublePendulumController object.

        Inputs:
        Double Pendulum object, double_pendulum: DoublePendulum
        Estimate of double_pendulum's first mass in kg, m_1_hat: float
        Estimate of double_pendulum's second mass in kg, m_2_hat: float
        Proportional contoller coefficients, K_p: numpy array (2,2)
        Derivative controller coefficient, K_d: numpy array (2, 2)
        Angle trajectory, r: float -> numpy array (2,)
        Angular velocity trajectory, r_dot: float ->  numpy array (2,) 
        Angular acceleration trajectory, r_ddot: float ->  numpy array (2,)
        
        """
        self.double_pendulum = DoublePendulum(m_1_hat, m_2_hat, double_pendulum.g, double_pendulum.l_1, double_pendulum.l_2, double_pendulum.a, double_pendulum.b)
        self.K_p = K_p
        self.K_d = K_d

        A = block([[zeros((2, 2)), identity(2)], [-self.K_p, -self.K_d]])
        Q = identity(4)
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.lambda_1 = max(eigvals(self.P))

        self.r, self.r_dot, self.r_ddot = r, r_dot, r_ddot

    def dVdx(self, x, t):
        e = x - concatenate((self.r(t), self.r_dot(t)))
        return 2 * dot(e, self.P)

    def LfV(self, x, t):
        """Compute Lie derivative, L_fV(x) = (dV / dx) * (f(x) - [r_dot(t), r_ddot(t)]).

        Outputs a float.

        Inputs:
        State, x: numpy array (2,)
        Time, t: float
        """
        return dot(self.dVdx(x, t), self.double_pendulum.drift(x) - concatenate((self.r_dot(t), self.r_ddot(t))))

    def LgV(self, x, t):
        """Compute Lie derivative, L_gV(x) = (dV / dx) * g(x)

        Outputs a float.

        Inputs:
        State, x: numpy array (2,)
        Time, t: float
        """

        return dot(self.dVdx(x, t), self.double_pendulum.act(x))

    def u(self, x, t):
        print(t)
        tol = 1e-6
        LgV = self.LgV(x, t)
        # Dual optimal solution
        if norm(LgV) < tol:
            lambda_star = 0
        else:
            lambda_star = (self.LfV(x, t) + self.V(x, t) / self.lambda_1) / (norm(LgV) ** 2)
            lambda_star = max(0, lambda_star)
        return -lambda_star * LgV
        

    def V(self, x, t):
        e = x - concatenate((self.r(t), self.r_dot(t)))
        return dot(e, dot(self.P, e))

    def dV(self, x, u, t):
        return self.LfV(x, t) + dot(self.LgV(x, t), u)
