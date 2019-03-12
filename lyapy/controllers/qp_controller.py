"""Quadratic program controller."""

from numpy import array, dot, identity, Inf, zeros
from numpy.linalg import norm

from ..lyapunov_functions import LearnedQuadraticControlLyapunovFunction
from .controller import Controller
from .util import solve_control_qp

class QPController(Controller):
    """Quadratic program controller.

    QP is

	  inf     1 / 2 * u'Pu + q'u + r + 1 / 2 * C * a'(P^-1)a * delta^2
	u, delta
                        + (u - u_prev)'H(u - u_prev)

	  s.t     a'u + b <= delta.

    If C is Inf, the slack, delta, is removed from the problem. Exception will
	then be raised if problem is infeasible. Previously computed control input
    is maintained as internal state, u_prev.

    Let n be the number of states, m be the number of inputs.

    Attributes:
    Control task output, output: Output
    Input size, m: int
    Cost function Hessian, P: numpy array (n,) * float -> numpy array (m, m)
    Cost function linear term, q: numpy array (n,) * float -> numpy array (m,)
    Cost function scalar term, r: numpy array (n,) * float -> float
    Constraint function linear term, a: numpy array (n,) * float -> numpy array (m,)
    Constraint function scalar term, b: numpy array (n,) * float -> float
    Slack weight, C: float
    Slack, delta: float
    Smoothness cost, H: numpy array (m, m)
    """

    def __init__(self, output, m, P=None, q=None, r=None, a=None, b=None, C=Inf, H=None):
        """Initialize a QPController object.

        Inputs:
        Control task output, output: Output
        Input size, m: int
        Cost function Hessian, P: numpy array (n,) * float -> numpy array (m, m)
        Cost function linear term, q: numpy array (n,) * float -> numpy array (m,)
        Cost function scalar term, r: numpy array (n,) * float -> float
        Constraint function linear term, a: numpy array (n,) * float -> numpy array (m,)
        Constraint function scalar term, b: numpy array (n,) * float -> float
        Slack weight, C: float
        Smoothness cost, H: numpy array (m, m)
        """

        Controller.__init__(self, output)

        if P is None:
            P = lambda x, t: identity(m)
        if q is None:
            q = lambda x, t: zeros(m)
        if r is None:
            r = lambda x, t: 0
        if a is None:
            a = lambda x, t: zeros(m)
        if b is None:
            b = lambda x, t: 0

        self.m = m
        self.P, self.q, self.r = P, q, r
        self.a, self.b = a, b
        self.C = C
        self.delta = None

        self.u_prev = None
        if H is None:
            H = zeros((m, m))
        self.H = H

    def u(self, x, t, update=True):

        m = self.m
        P, q, r = self.P(x, t), self.q(x, t), self.r(x, t)

        if self.u_prev is None:
            H = zeros((m, m))
        else:
            H = self.H
            P += H
            q += -dot(H, self.u_prev)
            r += 0.5 * dot(self.u_prev, dot(H, self.u_prev))

        a, b = self.a(x, t), self.b(x, t)
        C = self.C
        u_qp, self.delta = solve_control_qp(m, P, q, r, a, b, C)

        if update:
            self.u_prev = u_qp

        return u_qp

    def evaluate_slack(self, xs, ts):
        """Evaluate slack variables for given states and times.

        Let n be the number of states, T be the number states and times.

        Returns a numpy array (T,)

        Inputs:
        States, xs: numpy array (T, n)
        Times, ts: numpy array (T,)
        """

        self.reset()

        def slack(x, t):
            _ = self.u(x, t)
            return self.delta

        return array([slack(x, t) for x, t in zip(xs, ts)])

    def build_min_norm(quadratic_control_lyapunov_function, C=Inf, H=None):
        """Build a minimum norm controller for an affine dynamic output.

        QP is

          inf     1 / 2 * u'u + C * (decoupling)'(decoupling) * delta^2
        u, delta
                        + (u - u_prev)'H(u - u_prev)

          s.t     (decoupling)'u + drift <= -alpha * V + delta.

        If C is Inf, the slack, delta, is removed from the problem. Exception
        will then be raised if problem is infeasible.

        Outputs a QP Controller.

        Inputs:
        Quadratic CLF, quadratic_control_lyapunov_function: QuadraticControlLyapunovFunction
        Slack weight, C: float
        Smoothness cost, H: numpy array (m, m)
        """

        affine_dynamic_output = quadratic_control_lyapunov_function.output
        m = affine_dynamic_output.G.shape[-1]
        a = quadratic_control_lyapunov_function.decoupling
        alpha = quadratic_control_lyapunov_function.alpha
        V = quadratic_control_lyapunov_function.V
        b = lambda x, t: quadratic_control_lyapunov_function.drift(x, t) + alpha * V(x, t)
        return QPController(affine_dynamic_output, m, a=a, b=b, C=C, H=H)

    def build_aug(nominal_controller, m, quadratic_control_lyapunov_function, a, b, C=Inf, H=None):
        """Build a minimum norm augmenting controller for an affine dynamic output.

        QP is

          inf     1 / 2 * (u + u_c)'(u + u_c) + C * (V_decoupling + a)'(V_decoupling + a) * delta^2
        u, delta
                        + (u - u_prev)'H(u - u_prev)

          s.t     V_drift + V_decoupling * u_c + V_decoupling * u + a'(u + u_c) <= -alpha * V + delta.

        If C is Inf, the slack, delta, is removed from the problem. Exception
        will then be raised if problem is infeasible.

        Outputs a QP Controller.

        Inputs:
        Nominal controller, nominal_controller: Controller
        Input size, m: int
        Quadratic CLF: quadratic_control_lyapunov_function: QuadraticControlLyapunovFunction
        Modeled constraint linear term, a: numpy array (n,) * float -> numpy array (m,)
        Modeled constraint scalar term, b: numpy array (n,) * float -> float
        Slack weight, C: float
        Smoothness cost, H: numpy array (m, m)
        """

        learned_quadratic_control_lyapunov_function = LearnedQuadraticControlLyapunovFunction.build(quadratic_control_lyapunov_function, a, b)
        affine_dynamic_output = quadratic_control_lyapunov_function.output
        q = nominal_controller.u
        r = lambda x, t: (norm(q(x, t)) ** 2) / 2
        a = learned_quadratic_control_lyapunov_function.decoupling
        alpha = quadratic_control_lyapunov_function.alpha
        V = quadratic_control_lyapunov_function.V
        b = lambda x, t: learned_quadratic_control_lyapunov_function.drift(x, t) + dot(a(x, t), q(x, t)) + alpha * V(x, t)
        return QPController(affine_dynamic_output, m, q=q, r=r, a=a, b=b, C=C, H=H)

    def reset(self):
        self.u_prev = None
