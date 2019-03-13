"""Trainer object for Keras models."""

from numpy import zeros
from numpy.random import permutation

from .trainer import Trainer
from .util import connect_models, multi_layer_nn, TrainingLossThreshold

class KerasTrainer(Trainer):
    """Trainer object for Keras models.

    Let n be the number of states, m be the number of control inputs, s be the
    input vector size.

    Attributes:
    Function mapping state and time to model inputs, input: numpy array (n,) * float -> numpy array (s,)
    Lyapunov function, lyapunov_function: LyapunovFunction object
    Data subsample rate, subsample_rate: float
    Variable-step differentiator, diff: numpy array (T,) * numpy array (T,) -> numpy array (T - (L - 1) // 2,)
    Half of differentiator window size, half_diff_window: int
    Number of states, n: int
    Size of input vector, s: int
    Number of control inputs, m: int
    Hidden layer dimension: d_hidden: int
    List of Keras callbacks, callbacks: Keras Callback list
    Maximum number of epochs, max_epochs: int
    Fraction of data for a batch, batch_fraction: float
    Validation split, validation_split: float
    """

    def __init__(self, input, lyapunov_function, diff_window, subsample_rate, n, s, m, d_hidden, N_hidden=1, training_loss_threshold=1e-3, max_epochs=1000, batch_fraction=1, validation_split=0):
        """Initialize a KerasTrainer object.

        Inputs:
        Function mapping state and time to model inputs, input: numpy array (n,) * float -> numpy array (s,)
        Lyapunov function, lyapunov_function: LyapunovFunction object
        Data subsample rate, subsample_rate: float
        Differentiator window size, diff_window: int
        Number of states, n: int
        Size of input vector, s: int
        Number of control inputs, m: int
        Hidden layer dimension, d_hidden: int
        Number of hidden layers, N_hidden: int
        Training loss threshold, training_loss_threshold: float
        Maximum number of epochs, max_epochs: int
        Fraction of data for a batch, batch_fraction: float
        Validation split, validation_split: float
        """

        Trainer.__init__(self, input, lyapunov_function, diff_window, subsample_rate, n, s, m)
        self.d_hidden = d_hidden
        self.N_hidden = N_hidden
        self.callbacks = [TrainingLossThreshold(training_loss_threshold)]
        self.max_epochs = max_epochs
        self.batch_fraction = batch_fraction
        self.validation_split = validation_split

    def shuffle(self, data):
        """Shuffle data.

        Let N be the number of data points

        Outputs a numpy array (N, ...)

        Inputs:
        Data array, data: numpy array (N, ...)
        """

        N = len(data[0])
        perm = permutation(N)
        return tuple(_data[perm] for _data in data)

    def init_log(self):
        return ((zeros((0, self.m)), zeros(0), zeros(0)), ([], []))

    def update_log(self, log, train_data, a, b, deltas):
        _, _, _, inputs, _, _, _ = train_data
        a_predicts = a.predict(inputs)
        b_predicts = b.predict(inputs)[:, 0]
        data_log, (a_log, b_log) = log
        data_log = self.aggregate(data_log, (a_predicts, b_predicts, deltas))
        a_log.append(a)
        b_log.append(b)
        log = (data_log, (a_log, b_log))
        return log

    def fit(self, train_data):
        _, _, decouplings, inputs, u_noms, u_perts, V_dot_rs = self.shuffle(train_data)
        batch_size = int(len(V_dot_rs) * self.batch_fraction)

        a = multi_layer_nn(self.s, self.d_hidden, self.N_hidden, (self.m,))
        b = multi_layer_nn(self.s, self.d_hidden, self.N_hidden, (1,))
        model = connect_models(a, b)
        model.compile('adam', 'mean_absolute_error')

        model.fit([decouplings, inputs, u_noms, u_perts], V_dot_rs, callbacks=self.callbacks, epochs=self.max_epochs, batch_size=batch_size, validation_split=self.validation_split)

        return a, b
