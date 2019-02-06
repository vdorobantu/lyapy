"""Proportional-derivative controller for PD outputs."""

from numpy import dot

from .controller import Controller

class PDController(Controller):
    """Proportional-derivative controller for PD outputs.

    Let n be the number of states, m be the number of inputs, k be the number of
    outputs.

    Attributes:
    Control task output, output: PDOutput
    Proportional gain matrix, K_p: numpy array (m, k)
    Derivative gain matrix, K_d: numpy array (m, k)
    """

    def __init__(self, pd_output, K_p, K_d):
        """Initialize a PDController object.

        Inputs:
        Control task output, pd_output: PDOutput
        Proportional gain matrix, K_p: numpy array (m, k)
        Derivative gain matrix, K_d: numpy array (m, k)
        """
        Controller.__init__(self, pd_output)
        self.K_p = K_p
        self.K_d = K_d

    def u(self, x, t):
        e_p = self.output.proportional(x, t)
        e_d = self.output.derivative(x, t)
        return dot(self.K_p, e_p) + dot(self.K_d, e_d)
