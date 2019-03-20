"""Utilities for control design"""

from numpy import array, dot, identity, Inf, max, min, ones, reshape, tensordot, tile, zeros
from numpy.linalg import norm, solve
from numpy.random import uniform

from .controller import Controller

def solve_control_qp(m, P=None, q=None, r=0, a=None, b=0, C=Inf):
	"""Solve quadratic program that specifies control action and constraint slack.

	QP is

	  inf     1 / 2 * u'Pu + q'u + r + 1 / 2 * C * a'(P^-1)a * delta^2
	u, delta

	  s.t     a'u + b <= delta.

	If C is Inf, the slack, delta, is removed from the problem. Exception will
	then be raised if problem is infeasible.

	Let m be the number of inputs.

	Outputs a numpy array (m,) * float.

	Inputs:
	Input size, m: int
	Cost function Hessian, P: numpy array (m, m)
	Cost function linear term, q: numpy array (m,)
	Cost function scalar term, r: float
	Constraint function linear term, a: numpy array (m,)
	Constraint function scalar term, b: float
	Slack weight, C: float
	"""

	if P is None:
		P = identity(m)
	if q is None:
		q = zeros(m)
	if a is None:
		a = zeros(m)

	P_inv_q = solve(P, q)
	quad = dot(a, solve(P, a))

	if any(a != 0):
			lambda_cons = max([0, (b - dot(P_inv_q, a)) / (quad + 1 / (C * quad))])
			u = -solve(P, q + lambda_cons * a)
			delta = lambda_cons / (C * quad)
	else:
		u = -P_inv_q
		if C == Inf:
			if b > 0:
				raise(Exception('QP infeasible'))
			delta = None
		else:
			delta = max([0, b])

	return u, delta

class CombinedController(Controller):
	"""Linear combination of controllers.

	Computes u(x, t) = w_1 * u_1(x, t) + ... + w_M * u_M(x, t).

	Let M be the number of controllers to combine.

	Attributes:
	Control task output, output: Output
	Controller list, controllers: Controller list
	Controller weight array, weights: numpy array (M,)
	"""

	def __init__(self, controllers, weights, output=None):
		"""Initialize a CombinedController object.

		Inputs:
		Control task output, output: Output
		Controller list, controllers: Controller list
		Controller weight array, weights: numpy array (M,)
		"""

		if output is None:
			output = controllers[0].output
		Controller.__init__(self, output)
		self.controllers = controllers
		self.weights = weights

	def u(self, x, t, update=True):
		us = array([controller.u(x, t, update) for controller in self.controllers]).T
		return dot(us, self.weights)

	def evaluate(self, xs, ts):
		self.reset()
		us = array([controller.evaluate(xs, ts) for controller in self.controllers])
		us = tensordot(self.weights, us, [0, 0])
		return us

	def reset(self):
		for controller in self.controllers:
			controller.reset()


class ConstantController(Controller):
	"""Constant control action.

	Let m be the number of control inputs.

	Attributes:
	Control task output, output: Output
	Constant control action: numpy array (m,)
	"""

	def __init__(self, output, u_0):
		"""Initialize a ConstantController object.

		Inputs:
		Control task output, output: Output
		Constant control action, u_0: numpy array (m,)
		"""
		Controller.__init__(self, output)
		self.u_0 = u_0

	def u(self, x, t, update=True):
		return self.u_0

class PerturbingController(Controller):
	"""Predetermined time-based controller, scaled by norm of baseline controller.

	Computes u(x, t) = u_predetermined(t) * (offset + scaling
	* norm(u_baseline(x, t))).

	Let m be the number of control inputs.

	Attributes:
	Control task output, output: Output
	Baseline controller, controller: Controller
	Time-control dictionary, us: float * numpy array (m,) dict
	Norm of baseline controller scaling, scaling: float
	Norm of baseline controller offset, offset: float
	"""

	def __init__(self, output, controller, ts, us, scaling=1, offset=0):
		"""Initialize a PerturbingController object.

		Inputs:
		Control task output, output: Output
		Baseline controller, controller: Controller
		Time history, ts: float list
		Control history, us: numpy array (m,) list
		Norm of baseline controller scaling, scaling: float
		Norm of baseline controller offset, offset: float
		"""

		Controller.__init__(self, output)
		self.controller = controller
		self.us = {t:u for t, u in zip(ts, us)}
		self.scaling = scaling
		self.offset = offset

	def _u(self, x, t, update):
		"""Compute underlying control input with/without update flag.

		Returns a numpy array (m,)

		Inputs:
		State, x: numpy array (n,)
		Time, t: float
		Update flag, update: bool
		"""

		return self.us[t] * (self.offset + self.scaling * norm(self.controller.u(x, t, update)))

	def u(self, x, t, update=True):
		return self._u(x, t, False)

	def evaluate(self, xs, ts):
		self.controller.reset()
		return array([self._u(x, t, True) for x, t in zip(xs, ts)])

	def build(output, controller, t_eval, m, subsample_rate, width, scaling=1, offset=0):
		"""Build a PerturbingController object.

		Let T be the number of simulation time points.

		Perturbations are of the form

		xi * (offset + scaling * norm(u(x, t)))

		for xi element-wise uniformly drawn from [-width, width], and u a
		nominal controller.

		Outputs a PerturbingController.

		Inputs:
		Control task output, output: Output
		Nominal controller, controller: Controller
		Simulation time points, t_eval: numpy array (T,)
		Number of control inputs, m: int
		Subsample rate: subsample_rate: int
		Half-width of uniform distribution, width: float
		Norm of baseline controller scaling, scaling: float
		Norm of baseline controller offset, offset: float
		"""

		perturbations = uniform(-width, width, (len(t_eval) // subsample_rate, m))
		perturbations = reshape(tile(perturbations, [1, subsample_rate]), (-1, m))
		return PerturbingController(output, controller, t_eval, perturbations, scaling, offset)

class SaturationController(Controller):
	"""Controller incorporating element-wise saturation limits.

	Let m be the number of inputs.

	Attributes:
	Control task output, output: Output
	Baseline controller, controller: Controller
	Lower element-wise control bounds, lower_bounds: numpy array (m,)
	Upper element-wise control bounds, upper_bounds: numpy array (m,)
	"""

	def __init__(self, output, controller, m, lower_bounds=None, upper_bounds=None):
		"""Initialize a SaturationController object.

		Inputs:
		Control task output, output: Output
		Baseline controller, controller: Controller
		Number of control inputs, m: int
		Lower element-wise control bounds, lower_bounds: numpy array (m,)
		Upper element-wise control bounds, upper_bounds: numpy array (m,)
		"""

		Controller.__init__(self, output)

		if lower_bounds is None:
			lower_bounds = -Inf * ones(m)
		if upper_bounds is None:
			upper_bounds = Inf * ones(m)

		self.controller = controller
		self.lower_bounds = lower_bounds
		self.upper_bounds = upper_bounds

	def saturate(self, u):
		"""Saturate control input.

		Returns a numpy array (m,)

		Inputs:
		Unsaturated control input, u: numpy array (m,)
		"""

		return min([max([self.lower_bounds, u], axis=0), self.upper_bounds], axis=0)

	def u(self, x, t, update=True):
		return self.saturate(self.controller.u(x, t, update))

class AdaptiveController(Controller):
	def __init__(self, controller):
		Controller.__init__(self, controller.output)
		self.controller = controller

	def u(self, x, t, update=True):
		u_0 = self.controller.u(x, t, update=update)
		if update:
			self.output.update_params(x, u_0, t)
		return u_0

	def reset(self):
		self.output.reset()

	def evaluate_params(self, xs, ts):
		self.reset()

		def params(x, t):
			_ = self.u(x, t)
			return self.output.params

		return array([params(x, t) for x, t in zip(xs, ts)])
