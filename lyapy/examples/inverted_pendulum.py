"""Inverted pendulum example."""

from cvxpy import Maximize, Problem, Variable
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, xlabel
from numpy import arange, array, concatenate, dot, exp, identity, ones, savez, sin, zeros
from numpy.linalg import norm
from numpy.random import uniform, multivariate_normal
from scipy.io import loadmat

from ..controllers import AdaptiveController, PDController, QPController, CombinedController
from ..learning import decay_widths, evaluator, KerasTrainer, sigmoid_weighting, SimulationHandler
from ..lyapunov_functions import QuadraticControlLyapunovFunction
from ..outputs import AdaptiveOutput, RoboticSystemOutput
from ..systems import AffineControlSystem

class InvertedPendulum(AffineControlSystem):
	"""Inverted pendulum model.

	States are x = (theta, theta_dot), where theta is the angle of the pendulum
	in rad clockwise from upright and theta_dot is the angular rate of the
	pendulum in rad/s clockwise. The input is u = (tau), where tau is torque in
	N * m, applied clockwise at the base of the pendulum.

	Attributes:
	Mass (kg), m: float
	Gravitational acceleration (m/s^2), g: float
	Length (m), l: float
	"""

	def __init__(self, m, g, l, nl_damping=False):
		"""Initialize an InvertedPendulum object.

		Inputs:
		Mass (kg), m: float
		Gravitational acceleration (m/s^2), g: float
		Length (m), l: float
		"""

		AffineControlSystem.__init__(self)
		self.m, self.g, self.l = m, g, l
		self.damping = lambda theta: 1 / (m * (l ** 2))
		# self.damping = lambda theta: 0
		if nl_damping:
			# self.damping = lambda theta: 1 / (m * (l ** 2))
			self.damping = lambda theta: (1 - exp(-1 * (theta ** 2))) / (m * (l ** 2))
			# self.damping = lambda theta: sin(theta)

	def drift(self, x):
		theta, theta_dot = x
		return array([theta_dot, self.g / self.l * sin(theta) - self.damping(theta) * theta_dot])

	def act(self, x):
		return array([[0], [1 / (self.m * (self.l ** 2))]])

class InvertedPendulumOutput(RoboticSystemOutput):
	"""Inverted pendulum output.

	Outputs are eta = (theta - theta_d(t), theta_dot - theta_dot_d(t)), where
	theta and theta_dot are as specified in the InvertedPendulum class. theta_d
	and theta_dot_d are interpolated from a sequence of desired states, x_d.

	Attributes:
	List of relative degrees, vector_relative_degree: int list
	Permutation indices, permutation_idxs: numpy array (2,)
	Reverse permutation indices, reverse_permutation_idxs: numpy array (2,)
	Indices of k outputs when eta in block form, relative_degree_idxs: numpy array (1,)
	Indices of permutation into form with highest order derivatives in block, blocking_idxs: numpy array (2,)
	Indices of reverse permutation into form with highest order derivatives in block, unblocking_idxs: numpy array (2,)
	Linear output update matrix after decoupling inversion and drift removal, F: numpy array (2, 2)
	Linear output actuation matrix after decoupling inversion and drift removal, G: numpy array (2, 1)
	"""

	def __init__(self, inverted_pendulum, ts, x_ds):
		"""Initialize an InvertedPendulumOutput object.

		Let T be the number of time points specified.

		Inputs:
		Inverted pendulum system, inverted_pendulum: InvertedPendulum
		Time points, ts: numpy array (T,)
		Desired state points, x_ds: numpy array (T, n)
		"""

		RoboticSystemOutput.__init__(self, 1)
		self.inverted_pendulum = inverted_pendulum
		self.r, self.r_dot = self.interpolator(ts, x_ds[:, :1], x_ds[:, 1:])

	def eta(self, x, t):
		return x - self.r(t)

	def drift(self, x, t):
		return self.inverted_pendulum.drift(x) - self.r_dot(t)

	def decoupling(self, x, t):
		return self.inverted_pendulum.act(x)


from numpy import multiply, tensordot
outer = multiply.outer

class AdaptiveInvertedPendulumOutput(RoboticSystemOutput, AdaptiveOutput):
	def __init__(self, inverted_pendulum, gain, ts, x_ds, init_params=None, init_t=None):
		RoboticSystemOutput.__init__(self, 1)
		AdaptiveOutput.__init__(self, init_params, init_t)
		self.inverted_pendulum = inverted_pendulum
		self.gain = gain
		self.lyapunov_function = None
		self.r, self.r_dot = self.interpolator(ts, x_ds[:, :1], x_ds[:, 1:])

	def eta(self, x, t):
		return x - self.r(t)

	def dist_matrix(self, x, t):
		return array([[0, 0], [self.inverted_pendulum.g * sin(x[0]), 0]])

	def dist_tensor(self, x, t):
		return outer(outer(array([0, 1]), array([1])), array([0, 1]))

	def drift(self, x, t):
		return self.inverted_pendulum.drift(x) - self.r_dot(t) + dot(self.dist_matrix(x, t), self.params)

	def decoupling(self, x, t):
		return self.inverted_pendulum.act(x) + dot(self.dist_tensor(x, t), self.params)

	def update_params(self, x, u, t):
		tau = dot(self.lyapunov_function.grad_V(x, t), tensordot(self.dist_tensor(x, t), u, axes=[1, 0]) + self.dist_matrix(x, t))
		self.params = self.params + dot(self.gain, tau) * (t - self.t)
		self.t = t

	def set_lyapunov_function(self, lyapunov_function):
		self.lyapunov_function = lyapunov_function

# System parameters
m_hat, g, l_hat = 0.25, 9.81, 0.5 # Estimated parameters
delta = 0.25 # Max parameter variation
# m = uniform((1 - delta) * m_hat, (1 + delta) * m_hat) # True mass
# l = uniform((1 - delta) * l_hat, (1 + delta) * l_hat) # True length
# m, l = 0.2650362798572108, 0.5287463881925855
m, l = 0.20207737400995263, 0.576830978935558
system = InvertedPendulum(m_hat, g, l_hat) # Estimated system
system_true = InvertedPendulum(m, g, l, True) # Actual system

# Control design parameters
K_p = array([[-2.5]]) # PD controller P gain
K_d = array([[-0.375]]) # PD controller D gain
n = 2 # Number of states
m = 1 # Number of inputs
Q = identity(n) # Positive definite Q for CARE

# Simulation parameters
x_0 = array([1, 0]) # Initial condition
dt = 1e-3 # Time step
N = 5000 # Number of time steps
t_eval = dt * arange(N + 1) # Simulation time points

# Loading trajectory data
res = loadmat('./lyapy/trajectories/inverted_pendulum.mat')
t_ds = res['T_d'][:, 0] # Time points
x_ds = res['X_d'] # Desired state points

# System and control definitions
output = InvertedPendulumOutput(system, t_ds, x_ds)
pd_controller = PDController(output, K_p, K_d)
lyapunov_function = QuadraticControlLyapunovFunction.build_care(output, Q)
qp_controller = QPController.build_min_norm(lyapunov_function)

# Adaptive control definition
gain = identity(2)
init_params = zeros(2)
init_t = 0
adaptive_output = AdaptiveInvertedPendulumOutput(system, gain, t_ds, x_ds, init_params, init_t)
adaptive_lyapunov_function = QuadraticControlLyapunovFunction.build_care(adaptive_output, Q)
adaptive_output.set_lyapunov_function(adaptive_lyapunov_function)
adaptive_qp_controller = QPController.build_min_norm(adaptive_lyapunov_function)
adaptive_qp_controller = AdaptiveController(adaptive_qp_controller)

# # Input to models is state and nonzero component of Lyapunov function gradient
# # (sparsity comes from sparsity of system acutation matrix)
# input = lambda x, t: concatenate([x, lyapunov_function.grad_V(x, t)[-output.k:]])
# s = n + output.k
#
# # Learning loop
# num_episodes = 5 # Number of episodes of simulation/training
# loss_threshold = 1e-3 # Training loss threshold for an episode
# max_epochs = 5000 # Maximum number of training epochs per episode
# batch_fraction = 0.1
# validation_split = 0.1 # Percentage of training data withheld for validation
# subsample_rate = 20 # Number of time steps perturbations are held for
# width = 0.1 # Width of uniform distribution of perturbations
# scaling = 1
# offset = 0.1 # Perturbation offset
# d_hidden = 200 # Hidden layer dimension
# N_hidden = 1
# C = 1e3 # Augmenting controller slack weight
# weight_final = 0.99
# add_episodes = 0
#
# # Numerical differentiation filter
# diff_window = 3
#
# # Run experiments in simulation
# handler = SimulationHandler(system_true, output, pd_controller, m, lyapunov_function, x_0, t_eval, subsample_rate, input, C, scaling=scaling, offset=offset)
# # Set up episodic learning with Keras
# trainer = KerasTrainer(input, lyapunov_function, diff_window, subsample_rate, n, s, m, d_hidden, N_hidden, loss_threshold, max_epochs, batch_fraction, validation_split)
# weights = sigmoid_weighting(num_episodes, weight_final)
# widths = width * ones(num_episodes)
#
# # Train and build augmented controller
# a, b, train_data, log = trainer.run(handler, weights, widths)
# (a_predicts, b_predicts, deltas), (a_logs, b_logs) = log
# a = evaluator(input, a)
# b = evaluator(input, b, scalar_output=True)
# aug_controller = QPController.build_aug(pd_controller, m, lyapunov_function, a, b, C)
# total_controller = CombinedController([pd_controller, aug_controller], ones(2))
#
# # True output Lyapunov function object for comparison
# output_true = InvertedPendulumOutput(system_true, t_ds, x_ds)
# lyapunov_function_true = QuadraticControlLyapunovFunction.build_care(output_true, Q)

# Adaptive controller simulation
ts, xs = system_true.simulate(x_0, adaptive_qp_controller, t_eval)
params = adaptive_qp_controller.evaluate_params(xs, ts)

figure()
plot(ts, xs[:, 0], linewidth=2, label='$\\theta$')
plot(ts, xs[:, 1], linewidth=2, label='$\\dot{\\theta}$')
plot(t_ds, x_ds[:, 0], '--', linewidth=2, label='$\\theta_d$')
plot(t_ds, x_ds[:, 1], '--', linewidth=2, label='$\\dot{\\theta}_d$')
grid()
title('Adaptive Control Tracking', fontsize=16)
xlabel('Time (sec)', fontsize=16)
legend(fontsize=16)

figure()
plot(ts, params[:, 0], linewidth=2, label='$\\psi_1$')
plot(ts, params[:, 1], linewidth=2, label='$\\psi_2$')
grid()
title('Adaptive Control Parameters', fontsize=16)
xlabel('Time (sec)', fontsize=16)
legend(fontsize=16)

# Nominal QP controller simulation
ts, xs = system_true.simulate(x_0, qp_controller, t_eval)
x_qps = xs

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('QP Controller', fontsize=16)
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# PD controller simulation
ts, xs = system_true.simulate(x_0, pd_controller, t_eval)
x_pds = xs

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('PD Controller', fontsize=16)
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# # Augmented controller simulation
# ts, xs = system_true.simulate(x_0, total_controller, t_eval)
# x_augs = xs
#
# figure()
# plot(ts, xs, linewidth=2)
# plot(t_ds, x_ds, '--', linewidth=2)
# title('Augmented Controller', fontsize=16)
# legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
# grid()
#
# # Additional evaluation plots
# a_tests = array([a(x, t) for x, t in zip(xs, ts)])
# a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
# b_tests = array([b(x, t) for x, t in zip(xs, ts)])
# b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])
#
# figure()
# suptitle('Test data', fontsize=16)
# subplot(2, 1, 1)
# plot(a_tests, label='$\\widehat{a}$', linewidth=2)
# plot(a_trues, label='$a$', linewidth=2)
# legend(fontsize=16)
# grid()
# subplot(2, 1, 2)
# plot(b_tests, label='$\\widehat{b}$', linewidth=2)
# plot(b_trues, label='$b$', linewidth=2)
# legend(fontsize=16)
# grid()
#
# xs, ts, _, _, u_noms, u_perts, V_dot_rs = train_data
# a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
# b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])
# V_dot_r_trues = array([lyapunov_function_true.V_dot(x, u_nom + u_pert, t) - lyapunov_function.V_dot(x, u_nom, t) for x, u_nom, u_pert, t in zip(xs, u_noms, u_perts, ts)])
#
# # # # Calculating point-wise bound on the residual error
# # # def error_bound(L_A, L_b, A_infinity, b_infinity, data, grad_V, eta_jac, decoupling, a_hat, b_hat, controller):
# # # 	xs, ts, u_noms, u_perts, V_dot_rs = data
# # # 	m = u_noms.shape[1]
# # #
# # # 	a_hats = [a_hat(x_train, t_train) for x_train, t_train  in zip(xs, ts)]
# # # 	b_hats = [b_hat(x_train, t_train) for x_train, t_train  in zip(xs, ts)]
# # # 	grad_Vs = [grad_V(x_train, t_train) for x_train, t_train in zip(xs, ts)]
# # # 	eta_jacs = [eta_jac(x_train, t_train) for x_train, t_train in zip(xs, ts)]
# # # 	V_dot_r_hats = [dot(decoupling(x_train, t_train ), u_pert) + dot(a_hat.T, u_nom + u_pert) + b_hat for x_train, t_train, u_nom, u_pert, a_hat, b_hat in zip(xs, ts, u_noms, u_perts, a_hats, b_hats)]
# # #
# # # 	def opt(x, t):
# # # 		print(t)
# # #
# # # 		a = Variable(m)
# # # 		b = Variable(1)
# # # 		obj = Maximize(a * controller.u(x, t) + b)
# # # 		cons = []
# # #
# # # 		a_hat_test = a_hat(x, t)
# # # 		b_hat_test = b_hat(x, t)
# # # 		grad_V_test = grad_V(x, t)
# # # 		eta_jac_test = eta_jac(x, t)
# # #
# # # 		def opt_terms(a_hat_train, b_hat_train, grad_V_train, eta_jac_train, u_nom, u_pert, V_dot_r_hat, V_dot_r, x_train):
# # # 			u = u_nom + u_pert
# # # 			error_obs = abs(V_dot_r - V_dot_r_hat)
# # # 			error_model = abs(dot((a_hat_test - a_hat_train).T, u) + b_hat_test - b_hat_train)
# # # 			error_inf = norm(dot(grad_V_train, eta_jac_train) - dot(grad_V_test, eta_jac_test))
# # # 			error_inf = error_inf * (A_infinity * norm(u) + b_infinity)
# # # 			error_lip = min(norm(dot(grad_V_train, eta_jac_train)), norm(dot(grad_V_test, eta_jac_test)))*norm(x_train - x)
# # # 			error_lip = error_lip * (L_A * norm(u) + L_b)
# # # 			return concatenate([u, ones(1)]), error_obs + error_model + error_inf + error_lip
# # #
# # # 		zipped = zip(a_hats, b_hats, grad_Vs, eta_jacs, u_noms, u_perts, V_dot_r_hats, V_dot_rs, xs)
# # # 		terms = [opt_terms(*params) for params in zipped]
# # # 		linear, affine = zip(*terms)
# # # 		linear = array(linear)
# # # 		linear = concatenate([linear, -linear])
# # # 		affine = array(affine)
# # # 		affine = concatenate([affine, affine])
# # # 		cons = [linear[:, :-1] * a + linear[:, -1] * b <= affine]
# # #
# # # 		prob = Problem(obj, cons)
# # # 		prob.solve(solver='ECOS')
# # # 		return a.value, b.value
# # # 	return opt
# # #
# # # eta_jac = lambda x, t: identity(2)
# # # m_guess = (1 - delta) * m_hat
# # # l_guess = (1 - delta) * l_hat
# # # A_infinity = abs(1/(m_guess*(l_guess**2)) - 1/(m_hat*(l_hat**2)))
# # # b_infinity = abs(g*(1/l_guess - 1/l_hat))
# # # L_A = 0
# # # L_b = b_infinity
# # # data = xs, ts, u_noms, u_perts, V_dot_rs
# # #
# # # a_0 = lambda x, t: zeros(1)
# # # b_0 = lambda x, t: 0
# # #
# # # aug = error_bound(L_A, L_b, A_infinity, b_infinity, data, lyapunov_function.grad_V, eta_jac, lyapunov_function.decoupling, a, b, total_controller)
# # # qp = error_bound(L_A, L_b, A_infinity, b_infinity, data, lyapunov_function.grad_V, eta_jac, lyapunov_function.decoupling, a_0, b_0, qp_controller)
# # # pd = error_bound(L_A, L_b, A_infinity, b_infinity, data, lyapunov_function.grad_V, eta_jac, lyapunov_function.decoupling, a_0, b_0, pd_controller)
# # #
# # # from matplotlib.pyplot import colorbar, get_cmap, scatter, xlabel, ylabel
# # #
# # # reps = 10
# # # cov = (0.1 ** 2) * identity(n)
# # # x_samples = concatenate([multivariate_normal(x, cov, reps) for x in xs])
# # # t_samples = concatenate([ones(reps) * t for t in ts])
# # #
# # # print('Computing bounds for augmented controller...')
# # # maxs = [aug(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)]
# # # a_maxs, b_maxs = zip(*maxs)
# # # a_maxs = array(a_maxs)
# # # b_maxs = array(b_maxs)
# # # u_maxs = array([total_controller.u(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)])
# # # upper_bounds = array([dot(a_max, u_max) + b_max[0] for a_max, b_max, u_max in zip(a_maxs, b_maxs, u_maxs)])
# # # a_acts = array([lyapunov_function_true.decoupling(x_sample, t_sample) - lyapunov_function.decoupling(x_sample, t_sample) - a(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)])
# # # b_acts = array([lyapunov_function_true.drift(x_sample, t_sample) - lyapunov_function.drift(x_sample, t_sample) - b(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)])
# # # acts = array([dot(a_act, u_max) + b_act for a_act, b_act, u_max in zip(a_acts, b_acts, u_maxs)])
# # #
# # # savez('./output/aug_data.npz', samples=x_samples, upper_bounds=upper_bounds, acts=acts, xs=x_augs)
# # #
# # # print('Computing bounds for QP controller...')
# # # maxs = [qp(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)]
# # # a_maxs, b_maxs = zip(*maxs)
# # # a_maxs = array(a_maxs)
# # # b_maxs = array(b_maxs)
# # # u_maxs = array([qp_controller.u(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)])
# # # upper_bounds = array([dot(a_max, u_max) + b_max[0] for a_max, b_max, u_max in zip(a_maxs, b_maxs, u_maxs)])
# # # a_acts = array([lyapunov_function_true.decoupling(x_sample, t_sample) - lyapunov_function.decoupling(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)])
# # # b_acts = array([lyapunov_function_true.drift(x_sample, t_sample) - lyapunov_function.drift(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)])
# # # acts = array([dot(a_act, u_max) + b_act for a_act, b_act, u_max in zip(a_acts, b_acts, u_maxs)])
# # #
# # # savez('./output/qp_data.npz', samples=x_samples, upper_bounds=upper_bounds, acts=acts, xs=x_qps)
# #
# # # print('Computing bounds for PD controller...')
# # # maxs = [pd(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)]
# # # a_maxs, b_maxs = zip(*maxs)
# # # a_maxs = array(a_maxs)
# # # b_maxs = array(b_maxs)
# # # u_maxs = array([pd_controller.u(x_sample, t_sample) for x_sample, t_sample in zip(x_samples, t_samples)])
# # # upper_bounds = array([dot(a_max, u_max) + b_max[0] for a_max, b_max, u_max in zip(a_maxs, b_maxs, u_maxs)])
# # #
# # # savez('./output/pd_data.npz', samples=x_samples, upper_bounds=upper_bounds, xs=x_pds)
# #
# # # figure()
# # # title('Training data upper bounds', fontsize=16)
# # # bounds = scatter(x_samples[:, 0], x_samples[:, 1], c=upper_bounds, cmap=get_cmap('jet'))
# # # colorbar(bounds)
# # # plot(x_ds[:, 0], x_ds[:, 1], '--k', linewidth=2)
# # # xlabel('$\\theta$', fontsize=16)
# # # ylabel('$\\dot{\\theta}$', fontsize=16)
# # # grid()
# #
# # # t_ds = t_eval[::subsample_rate]
# # # x_ds = array([output.r(t) for t in t_ds])
# # # maxs = [opt(x_d, t_d) for x_d, t_d in zip(x_ds, t_ds)]
# # # a_maxs, b_maxs = zip(*maxs)
# # # a_maxs = array(a_maxs)
# # # b_maxs = array(b_maxs)
# # # u_maxs = array([total_controller.u(x_d, t_d) for x_d, t_d in zip(x_ds, t_ds)])
# # # upper_bounds = array([dot(a_max, u_max) + b_max[0] for a_max, b_max, u_max in zip(a_maxs, b_maxs, u_maxs)])
# # #
# # # figure()
# # # title('Desired trajectory upper bounds', fontsize=16)
# # # bounds = scatter(x_ds[:, 0], x_ds[:, 1], c=upper_bounds, cmap=get_cmap('jet'))
# # # colorbar(bounds)
# # # grid()
# #
# figure()
# suptitle('Episode data', fontsize=16)
# subplot(2, 1, 1)
# title('State data', fontsize=16)
# plot(xs, linewidth=2)
# legend(['$\\theta$', '$\\dot{\\theta}$'], fontsize=16)
# grid()
# subplot(2, 1, 2)
# title('Control data', fontsize=16)
# plot(u_noms + u_perts, label='$u$', linewidth=2)
# legend(fontsize=16)
# grid()
#
# figure()
# plot(V_dot_rs, linewidth=2, label='$\\widehat{\\dot{V}}_r$')
# plot(V_dot_r_trues, '--', linewidth=2, label='$\\dot{V}_r$')
# title('Numerical differentiation', fontsize=16)
# legend(fontsize=16)
# grid()
#
# figure()
# suptitle('Training results', fontsize=16)
# subplot(2, 1, 1)
# plot(a_predicts, label='$\\widehat{a}$', linewidth=2)
# plot(a_trues, '--', label='$a$', linewidth=2)
# legend(fontsize=16)
# grid()
# subplot(2, 1, 2)
# plot(b_predicts, label='$\\widehat{b}$', linewidth=2)
# plot(b_trues, '--', label='$b$', linewidth=2)
# legend(fontsize=16)
# grid()
#
print(system_true.m, system_true.l)

show()
