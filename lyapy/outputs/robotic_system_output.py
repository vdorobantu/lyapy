from numpy import arange, array, concatenate, dot, ones, reshape, where
from numpy.linalg import solve

from .feedback_linearizable_output import FeedbackLinearizableOutput
from .pd_output import PDOutput

class RoboticSystemOutput(FeedbackLinearizableOutput, PDOutput):
    """Base class for robotic system outputs.

    Override eta, drift, decoupling.

    A robotic system is a system with
    states of the form x = (q, q_dot) with coordinates q in configuration space
    Q (subset of R^n) and coordinate rates q_dot in R^n. Additionally, the
    output is defined as eta = ( y(q) - y_d(t), d/dt (y(q) - y_d(t)) ) with
    y: Q --> R^k and time-based desired trajectory y_d : R --> R^k. y - y_d is
    the proportional component, d/dt (y - y_d) is the derivative component.

    Let n be the number of states, k be the number of outputs.

    Attributes:
    List of relative degrees, vector_relative_degree: int list
    Permutation indices, permutation_idxs: numpy array (2 * k,)
    Reverse permutation indices, reverse_permutation_idxs: numpy array (2 * k,)
    Indices of k outputs when eta in block form, relative_degree_idxs: numpy array (k,)
    Indices of permutation into form with highest order derivatives in block, blocking_idxs: numpy array (2 * k,)
    Indices of reverse permutation into form with highest order derivatives in block, unblocking_idxs: numpy array (2 * k,)
    Linear output update matrix after decoupling inversion and drift removal, F: numpy array (2 * k, 2 * k)
    Linear output actuation matrix after decoupling inversion and drift removal, G: numpy array (2 * k, k)
    """

    def __init__(self, k):
        """Initialize a RoboticSystemOutput.

        Inputs:
        Number of outputs, k: int
        """

        self.k = k
        vector_relative_degree = [2] * k
        permutation_idxs = reshape(array([arange(k), k + arange(k)]).T, -1)
        FeedbackLinearizableOutput.__init__(self, vector_relative_degree, permutation_idxs)

    def proportional(self, x, t):
        return self.eta(x, t)[:self.k]

    def derivative(self, x, t):
        return self.eta(x, t)[-self.k:]

    def interpolator(self, ts, y_ds, y_d_dots):
        """Generate functions which interpolate y_d at specified times.

        The two functions approximate (y_d, y_d_dot) and
        (y_d_dot, y_d_ddot). The interpolation assigns a cubic polynomial to
        each interval of adjacent time points.

        Outputs a (float -> numpy array (2 * k,)) * (float -> numpy array (2 * k,))

        Let T be the number of time points used.

        Inputs:
        Time points, ts: numpy array (T,)
        Specified y_d points, y_ds: numpy array (T, k)
        Specified y_d_dot points, y_d_dots: numpy array (T, k)
        """

        def interpolate(t):
            before, = where(ts <= t)
            after, = where(ts > t)

            if len(after) == 0:
                idx_0 = before[-2]
                idx_1 = before[-1]
            else:
                idx_0 = before[-1]
                idx_1 = after[0]

            t_0, y_d_0, y_d_dot_0 = ts[idx_0], y_ds[idx_0], y_d_dots[idx_0]
            t_1, y_d_1, y_d_dot_1 = ts[idx_1], y_ds[idx_1], y_d_dots[idx_1]

            A = array([
                [t_0 ** 3, t_0 ** 2, t_0, 1],
                [t_1 ** 3, t_1 ** 2, t_1, 1],
                [3 * (t_0 ** 2), 2 * t_0, 1, 0],
                [3 * (t_1 ** 2), 2 * t_1, 1, 0]
            ])

            bs = array([y_d_0, y_d_1, y_d_dot_0, y_d_dot_1])

            alphas_0 = solve(A, bs)
            alphas_1 = array([3 * alphas_0[0], 2 * alphas_0[1], alphas_0[2]])
            alphas_2 = array([2 * alphas_1[0], alphas_1[1]])

            ts_0 = t ** arange(3, -1, -1)
            ts_1 = ts_0[1:]
            ts_2 = ts_1[1:]

            y_d = dot(ts_0, alphas_0)
            y_d_dot = dot(ts_1, alphas_1)
            y_d_ddot = dot(ts_2, alphas_2)

            return concatenate([y_d, y_d_dot, y_d_ddot])

        def r(t):
            return interpolate(t)[:(2 * self.k)]

        def r_dot(t):
            return interpolate(t)[-(2 * self.k):]

        return r, r_dot
