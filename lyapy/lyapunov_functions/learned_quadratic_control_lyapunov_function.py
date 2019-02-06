"""Class for Quadratic CLFs, augmented with residual drift and decoupling models."""

from .quadratic_control_lyapunov_function import QuadraticControlLyapunovFunction

class LearnedQuadraticControlLyapunovFunction(QuadraticControlLyapunovFunction):
    """Class for Quadratic CLFs, augmented with residual drift and decoupling models.

    Attributes:
    Control task output, output: AffineDynamicOutput
    Positive definite matrix, P: numpy array (p, p)
    Convergence rate, alpha: float
    Residual decoupling model, a: numpy array (n,) * float -> numpy array (m,)
    Residual drift model, b: numpy array (n,) * float -> float
    """

    def __init__(self, affine_dynamic_output, P, alpha, a, b):
        """Initialize a LearnedQuadraticControlLyapunovFunction object.

        Inputs:
        Control task output, affine_dynamic_output: AffineDynamicOutput
        Positive definite matrix, P: numpy array (p, p)
        Convergence rate, alpha: float
        Residual decoupling model, a: numpy array (n,) * float -> numpy array (m,)
        Residual drift model, b: numpy array (n,) * float -> float
        """

        QuadraticControlLyapunovFunction.__init__(self, affine_dynamic_output, P, alpha)
        self.a = a
        self.b = b

    def drift(self, x, t):
        return super().drift(x, t) + self.b(x, t)

    def decoupling(self, x, t):
        return super().decoupling(x, t) + self.a(x, t)

    def build(quadratic_control_lyapunov_function, a, b):
        """Build a learned quadratic CLF from a quadratic CLF an residual drift and decoupling models.

        Outputs a LearnedQuadraticControlLyapunovFunction.

        Inputs:
        Nominal quadratic CLF, quadratic_control_lyapunov_function: QuadraticControlLyapunovFunction
        Residual decoupling model, a: numpy array (n,) * float -> numpy array (m,)
        Residual drift model, b: numpy array (n,) * float -> float
        """

        affine_dynamic_output = quadratic_control_lyapunov_function.output
        P = quadratic_control_lyapunov_function.P
        alpha = quadratic_control_lyapunov_function.alpha
        return LearnedQuadraticControlLyapunovFunction(affine_dynamic_output, P, alpha, a, b)
