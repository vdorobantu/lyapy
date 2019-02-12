"""Class for handling simulated experiments."""

from numpy import array, Inf, ones

from .handler import Handler
from .util import evaluator
from ..controllers import CombinedController, PerturbingController, QPController

class SimulationHandler(Handler):
    """Class for handling simulated experiments.

    Let n be the number of states, m be the number of control inputs, s be the
    size of the input vector. Let T be the number of simulation time points.

    Attributes:
    Simulation system, system: System
    Nominal controller, controller: Controller
    Number of control inputs, m: int
    Lyapunov function, lyapunov_function: LyapunovFunction
    Initial condition, x_0: numpy array (n,)
    Simulation times, t_eval: numpy array (T,)
    Function mapping state and time to model inputs, input: numpy array (n,) * float -> numpy array (s,)
    Slack weight, C: int
    PerturbingController generator, gen_pert: Controller -> PerturbingController
    """

    def __init__(self, system, output, controller, m, lyapunov_function, x_0, t_eval, subsample_rate, input, C=Inf, scaling=1, offset=0):
        """Initialize a SimulationHandler object.

        Inputs:
        Simulation system, system: System
        Control task output, output: Output
        Nominal controller, controller: Controller
        Number of control inputs, m: int
        Lyapunov function, lyapunov_function: LyapunovFunction
        Initial condition, x_0: numpy array (n,)
        Simulation times, t_eval: numpy array (T,)
        Subsample rate, subsample_rate: int
        Function mapping state and time to model inputs, input: numpy array (n,) * float -> numpy array (s,)
        Slack weight, C: int
        Norm of baseline controller scaling, scaling: float
		Norm of baseline controller offset, offset: float
        """

        self.system = system
        self.controller = controller
        self.m = m
        self.lyapunov_function = lyapunov_function
        self.x_0 = x_0
        self.t_eval = t_eval
        self.input = input
        self.C = C
        self.gen_pert = lambda nom_controller, width: PerturbingController.build(output, nom_controller, t_eval, m, subsample_rate, width, scaling, offset)

    def run(self, weight, width, a=None, b=None):
        if a is None or b is None:
            nom_controller = self.controller
        else:
            a = evaluator(self.input, a)
            b = evaluator(self.input, b, scalar_output=True)
            aug_controller = QPController.build_aug(self.controller, self.m, self.lyapunov_function, a, b, self.C)
            nom_controller = CombinedController([self.controller, aug_controller], array([1, weight]))

        pert_controller = self.gen_pert(nom_controller, width)
        total_controller = CombinedController([nom_controller, pert_controller], ones(2))

        ts, xs = self.system.simulate(self.x_0, total_controller, self.t_eval)
        u_noms = array([nom_controller.u(x, t) for x, t in zip(xs, ts)])
        u_perts = array([pert_controller.u(x, t) for x, t in zip(xs, ts)])

        return xs, u_noms, u_perts, ts
