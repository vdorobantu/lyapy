"""Base class for running episodic experiments."""

from numpy import array, concatenate, zeros

from .util import differentiator, sigmoid_weighting

class Trainer:
    """Base class for running episodic experiments.

    Override fit.

    init_log and update_log can be overridden for custom processing and logging.

    Let n be the number of states, s be the input vector size, m be the number
    of control inputs. Let T be the number of data points in an experiment.

    Attributes:
    Function mapping state and time to model inputs, input: numpy array (n,) * float -> numpy array (s,)
    Lyapunov function, lyapunov_function: LyapunovFunction object
    Data subsample rate, subsample_rate: float
    Variable-step differentiator, diff: numpy array (T,) * numpy array (T,) -> numpy array (T - (L - 1) // 2,)
    Half of differentiator window size, half_diff_window: int
    Number of states, n: int
    Size of input vector, s: int
    Number of control inputs, m: int
    """

    def __init__(self, input, lyapunov_function, diff_window, subsample_rate, n, s, m):
        """Initialize a Trainer object.

        Inputs:
        Function mapping state and time to model inputs, input: numpy array (n,) * float -> numpy array (s,)
        Lyapunov function, lyapunov_function: LyapunovFunction object
        Data subsample rate, subsample_rate: float
        Differentiator window size, diff_window: int
        Number of states, n: int
        Size of input vector, s: int
        Number of control inputs, m: int
        """

        self.input = input
        self.lyapunov_function = lyapunov_function
        self.subsample_rate = subsample_rate
        self.diff = differentiator(diff_window)
        self.half_diff_window = (diff_window - 1) // 2
        self.n = n
        self.s = s
        self.m = m

    def process(self, exp_data):
        """Form input data, estimate labels and subsample.

        Let T be the number of experiment data points. Let L be the size of the
        differentiation filter, and let r be the subsample rate.

        Let T' = (T - L + 1) // r.

        Outputs a numpy array (T', n) * numpy array (T',) * numpy array (T', m) * numpy array (T', s) * numpy array (T', m) * numpy array (T', m) * numpy array (T',).

        Inputs:
        Experiment data, exp_data: numpy array (T, n) * numpy array (T, m) * numpy array (T, m) * numpy array (T,)
        """

        xs, u_noms, u_perts, ts = exp_data

        Vs = array([self.lyapunov_function.V(x, t) for x, t in zip(xs, ts)])
        V_dots = self.subsample(self.diff(Vs, ts))
        xs = self.trim_and_subsample(xs)
        ts = self.trim_and_subsample(ts)
        decouplings = array([self.lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
        inputs = array([self.input(x, t) for x, t in zip(xs, ts)])
        u_noms = self.trim_and_subsample(u_noms)
        u_perts = self.trim_and_subsample(u_perts)
        V_dot_ds = array([self.lyapunov_function.V_dot(x, u_nom, t) for x, u_nom, t in zip(xs, u_noms, ts)])
        V_dot_rs = V_dots - V_dot_ds

        return xs, ts, decouplings, inputs, u_noms, u_perts, V_dot_rs

    def aggregate(self, acc, data):
        """Add more data to an accumulator.

        Let k_1 be the size of data in the accumulator, and let k_2 be the size
        of new data.

        Outputs a numpy array (k_1 + k_2, ...) * ... * numpy array (k_1 + k_2, ...).

        Inputs:
        Accumulator, acc: numpy array (k_1, ...) * ... * numpy array (k_1, ...)
        New data, data: numpy array (k_2, ...) * ... * numpy array (k_2, ...)
        """

        return tuple(concatenate([_acc, _data]) for _acc, _data in zip(acc, data))

    def fit(self, train_data):
        """Create models that fit the training data.

        Models are a and b satisfying

        V_dot_r = lyapunov_function.decoupling(x, t) * u + a(x, t) * (u_c(x, t) + u) + b(x, t)

        Let T be the number of training data points.

        Outputs a model (R^n * R -> R^m) * model (R^n * R -> R)

        Inputs:
        Training data, train_data: numpy array (T, n) * numpy array (T,) * numpy array (T, m) * numpy array (T, s) * numpy array (T, m) * numpy array (T, m) * numpy array (T,)
        """

        pass

    def init_log(self):
        """Initialize log.

        Outputs a log tuple.
        """

        pass

    def update_log(self, log, train_data, a, b):
        """Update log.

        Let T be the number of training data points.

        Outputs a log tuple.

        Inputs:
        Previous log, log: log tuple
        Latest training data, train_data: numpy array (T, n) * numpy array (T,) * numpy array (T, m) * numpy array (T, s) * numpy array (T, m) * numpy array (T, m) * numpy array (T,)
        Decoupling model, a: model (R^n * R -> R^m)
        Drift model, b: model (R^n * R -> R)
        """

        return log

    def run(self, handler, num_episodes, weight_final):
        """Run episodic experiments and learn augmenting Lyapunov function drift and decoupling models.

        Let N be the number of episodes and let T be the number of data points
        per episode.

        Outputs a model (R^n * R -> R^m) * model (R^n * R -> R) * (numpy array (N * T, n) * numpy array (N * T,) * numpy array (N * T, m) * numpy array (N * T, s) * numpy array (N * T, m) * numpy array (N * T, m) * numpy array (N * T,)) * log tuple

        Inputs:
        Experiment handler, handler: Handler object
        Number of episodes, num_episodes: int
        Final augmenting controller weight, weight_final: float
        """

        weights = sigmoid_weighting(num_episodes, weight_final)

        a = None
        b = None
        xs = zeros((0, self.n))
        ts = zeros(0)
        decouplings = zeros((0, self.m))
        inputs = zeros((0, self.s))
        u_noms = zeros((0, self.m))
        u_perts = zeros((0, self.m))
        V_dot_rs = zeros(0)
        train_data = (xs, ts, decouplings, inputs, u_noms, u_perts, V_dot_rs)
        log = self.init_log()

        for episode, weight in enumerate(weights):
            print('EPISODE', episode)

            exp_data = handler.run(weight, a, b)
            _train_data = self.process(exp_data)
            train_data = self.aggregate(train_data, _train_data)
            a, b = self.fit(train_data)
            log = self.update_log(log, _train_data, a, b)

        return a, b, train_data, log

    def subsample(self, arr):
        """Subsample an array.

        Let T be the number of items in the array, and let L be the size of the
        differentiator window.

        Outputs a numpy array (T - L + 1, ...).

        Inputs:
        Array, arr: numpy array (T, ...)
        """

        return arr[::self.subsample_rate]

    def trim_and_subsample(self, arr):
        """Trim and subsample an array.

        Let T be the number of items in the array, let L be the size of the
        differentiator window, and let r be the subsample rate.

        Outputs a numpy array ((T - L + 1) // r, ...).

        Inputs:
        Array, arr: numpy array (T, ...)
        """

        return arr[self.half_diff_window:-self.half_diff_window:self.subsample_rate]
