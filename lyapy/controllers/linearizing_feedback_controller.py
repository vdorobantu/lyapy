"""Linearizing feedback controller for feedback linearizable outputs."""

from numpy import dot
from numpy.linalg import solve

from .controller import Controller

class LinearizingFeedbackController(Controller):
    """Linearizing feedback controller for feedback linearizable outputs.

    Let n be the number of states, k be the number of outputs, p be the output
    vector size.

    Attributes:
    Control task output, output: FeedbackLinearizableOutput
    Output drift function, drift: numpy array (n,) * float -> numpy array (p,)
    Output decoupling function, decoupling: numpy array (n,) * float -> numpy array (p, m)
    Output permuation function, permute: numpy array (p, ...) -> numpy array (p, ...)
    Output selection function, select: numpy array (p, ...) -> numpy array (k, ...)
    Auxiliary control gain matrix, K: numpy array (k, p)
    """

    def __init__(self, feedback_linearizable_output, K):
        """Initialize a LinearizingFeedbackController object.

        Inputs:
        Control task output, feedback_linearizable_output: FeedbackLinearizableOutput
        """
        Controller.__init__(self, feedback_linearizable_output)
        self.drift = feedback_linearizable_output.drift
        self.decoupling = feedback_linearizable_output.decoupling
        self.permute = feedback_linearizable_output.permute
        self.select = feedback_linearizable_output.select
        self.K = K

    def u(self, x, t):
        eta = self.output.eta(x, t)
        drift = self.select(self.permute(self.drift(x, t)))
        decoupling = self.select(self.permute(self.decoupling(x, t)))
        return solve(decoupling, -drift - dot(self.K, eta))
