"""Quadratic program controller."""

from cvxpy import Minimize, Problem, quad_form, square, Variable
from numpy import dot, identity, Inf, ones, zeros
from numpy.linalg import norm

from ..lyapunov_functions import LearnedQuadraticControlLyapunovFunction
from .controller import Controller
from .util import solve_control_qp

class QPController(Controller):
    """Quadratic program controller.

    QP is

	  inf     1 / 2 * u'Pu + q'u + r + 1 / 2 * C * delta^2
	u, delta

	  s.t     a'u + b <= delta.
              u_mins <= u <= u_maxs
    If C is Inf, the slack, delta, is removed from the problem.

    Let n be the number of states, m be the number of inputs.

    Attributes:
    Control task output, output: Output
    Current control, u_0: numpy array (m,)
    Current slack, delta_0: float
    Optimization problem, prob: numpy array (n,) * float -> cvxpy Problem
    """

    def __init__(self, output, m, P=None, q=None, r=None, a=None, b=None, C=None, u_mins=None, u_maxs=None):
        """Initialize a QPController object.

        Inputs:
        Control task output, output: Output
        Input size, m: int
        Cost function Hessian, P: numpy array (n,) * float -> numpy array (m, m)
        Cost function linear term, q: numpy array (n,) * float -> numpy array (m,)
        Cost function scalar term, r: numpy array (n,) * float -> float
        Constraint function linear term, a: numpy array (n,) * float -> numpy array (m,)
        Constraint function scalar term, b: numpy array (n,) * float -> float
        Slack weight, C: numpy array (n,) * float -> float
        Lower control bounds, u_mins: numpy array (n,) * float -> numpy array (m,)
        Upper control bounds, u_maxs: numpy array (n,) * float -> numpy array (m,)
        """

        Controller.__init__(self, output)

        if P is None:
            P = lambda x, t: identity(m)
        if q is None:
            q = lambda x, t: zeros(m)
        if r is None:
            r = lambda x, t: 0

        u = Variable(m)
        delta = Variable(1)
        base_cost = lambda x, t: 1 / 2 * quad_form(u, P(x, t)) + q(x, t) * u + r(x, t)

        if C is None:
            cost = base_cost
        else:
            cost = lambda x, t: base_cost(x, t) + 1 / 2 * C(x, t) * square(delta)

        obj = lambda x, t: Minimize(cost(x, t))

        if a is None:
            a = lambda x, t: zeros(m)
        if b is None:
            b = lambda x, t: 0

        base_cons = lambda x, t: [a(x, t) * u + b(x, t) <= delta]

        if u_mins is None:
            u_mins = lambda x, t: -Inf * ones(m)
        if u_maxs is None:
            u_maxs = lambda x, t: Inf * ones(m)

        cons = lambda x, t: base_cons(x, t) + [u_mins(x, t) <= u, u <= u_maxs(x, t)]

        self.u_0 = u
        self.delta_0 = delta
        self.prob = lambda x, t: Problem(obj(x, t), cons(x, t))

    def u(self, x, t):
        self.prob(x, t).solve(warm_start=True)
        return self.u_0.value

    def build_min_norm(quadratic_control_lyapunov_function, slack_weight=None, u_mins=None, u_maxs=None):
        """Build a minimum norm controller for an affine dynamic output.

        QP is

          inf     1 / 2 * u'u + slack_weight * (decoupling)'(decoupling) * delta^2
        u, delta
          s.t     (decoupling)'u + drift <= -alpha * V + delta.

        If slack_weight is Inf, the slack, delta, is removed from the problem.

        Outputs a QP Controller.

        Inputs:
        Quadratic CLF, quadratic_control_lyapunov_function: QuadraticControlLyapunovFunction
        Slack weight, slack_weight: float
        Lower control bounds: u_mins: numpy array (n,) * float -> numpy array (m,)
        Upper control bounds: u_maxs: numpy array (n,) * float -> numpy array (m,)
        """

        affine_dynamic_output = quadratic_control_lyapunov_function.output
        m = affine_dynamic_output.G.shape[-1]
        a = quadratic_control_lyapunov_function.decoupling
        alpha = quadratic_control_lyapunov_function.alpha
        V = quadratic_control_lyapunov_function.V
        b = lambda x, t: quadratic_control_lyapunov_function.drift(x, t) + alpha * V(x, t)

        if slack_weight is None:
            C = None
        else:
            C = lambda x, t: slack_weight * (norm(a(x, t)) ** 2)

        return QPController(affine_dynamic_output, m, a=a, b=b, C=C, u_mins=u_mins, u_maxs=u_maxs)

    def build_aug(nominal_controller, m, quadratic_control_lyapunov_function, a, b, slack_weight=None, u_mins=None, u_maxs=None):
        """Build a minimum norm augmenting controller for an affine dynamic output.

        QP is

          inf     1 / 2 * (u + u_c)'(u + u_c) + slack_weight * (V_decoupling + a)'(V_decoupling + a) * delta^2
        u, delta
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
        Slack weight, slack_weight: float
        Lower control bounds: u_mins: numpy array (n,) * float -> numpy array (m,)
        Upper control bounds: u_maxs: numpy array (n,) * float -> numpy array (m,)
        """

        learned_quadratic_control_lyapunov_function = LearnedQuadraticControlLyapunovFunction.build(quadratic_control_lyapunov_function, a, b)
        affine_dynamic_output = quadratic_control_lyapunov_function.output
        q = nominal_controller.u
        r = lambda x, t: (norm(q(x, t)) ** 2) / 2
        a = learned_quadratic_control_lyapunov_function.decoupling
        alpha = quadratic_control_lyapunov_function.alpha
        V = quadratic_control_lyapunov_function.V
        b = lambda x, t: learned_quadratic_control_lyapunov_function.drift(x, t) + dot(a(x, t), q(x, t)) + alpha * V(x, t)

        if slack_weight is None:
            C = None
        else:
            C = lambda x, t: slack_weight * (norm(a(x, t)) ** 2)

        return QPController(affine_dynamic_output, m, q=q, r=r, a=a, b=b, C=C, u_mins=u_mins, u_maxs=u_maxs)
