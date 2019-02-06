"""Base class for experiment handlers."""

class Handler:
    """Base class for experiment handlers.

    Let n be the number of states, m be the number of inputs.

    Override run.
    """

    def run(self, weight, a, b):
        """Run an experiment.

        Let T be the number of data points in an experiment.

        Outputs a numpy array (T, n) * numpy array (T, m) * numpy array (T, m) * numpy array (T,).

        Inputs:
        Weight of augmenting controller, weight: float
        Lyapunov function decoupling model, a: model (R^n * R -> R^m)
        Lyapunov function drift model, b: model (R^n * R -> R)
        """

        pass
