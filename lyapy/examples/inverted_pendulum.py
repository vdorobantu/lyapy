"""Inverted pendulum example."""

from matplotlib.pyplot import figure, grid, legend, plot, semilogy, show, subplot, suptitle, title
from numpy import arange, array, concatenate, identity, linspace, ones, reshape, sin, sqrt, tile, zeros
from numpy.random import permutation, uniform
from scipy.io import loadmat

from ..controllers import PDController, QPController, CombinedController, ConstantController, PerturbingController
from ..learning import connect_models, differentiator, evaluator, sigmoid_weighting, TrainingLossThreshold, two_layer_nn
from ..lyapunov_functions import QuadraticControlLyapunovFunction
from ..outputs import RoboticSystemOutput
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

	def __init__(self, m, g, l):
		"""Initialize an InvertedPendulum object.

		Inputs:
		Mass (kg), m: float
		Gravitational acceleration (m/s^2), g: float
		Length (m), l: float
		"""

		AffineControlSystem.__init__(self)
		self.m, self.g, self.l = m, g, l

	def drift(self, x):
		theta, theta_dot = x
		return array([theta_dot, self.g / self.l * sin(theta)])

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

# System parameters
m_hat, g, l_hat = 0.25, 9.81, 0.5 # Estimated parameters
delta = 0.5 # Max parameter variation
m = uniform((1 - delta) * m_hat, (1 + delta) * m_hat) # True mass
l = uniform((1 - delta) * l_hat, (1 + delta) * l_hat) # True length
system = InvertedPendulum(m_hat, g, l_hat) # Estimated system
system_true = InvertedPendulum(m, g, l) # Actual system

# Control design parameters
K_p = array([[-10]]) # PD controller P gain
K_d = array([[-1]]) # PD controller D gain
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

# Input to models is state and nonzero component of Lyapunov function gradient
# (sparsity comes from sparsity of system acutation matrix)
input = lambda x, t: concatenate([x, lyapunov_function.grad_V(x, t)[-output.k:]])
s = n + output.k

# Learning loop
num_episodes = 20 # Number of episodes of simulation/training
loss_threshold = 1e-4 # Training loss threshold for an episode
max_epochs = 5000 # Maximum number of training epochs per episode
validation_split = 0.1 # Percentage of training data withheld for validation
subsample_rate = 20 # Number of time steps perturbations are held for
width = 0.1 # Width of uniform distribution of perturbations
offset = 0.1 # Perturbation offset
d_hidden = 200 # Hidden layer dimension
C = 1e3 # Augmenting controller slack weight

# Numerical differentiation filter
diff_window = 3
diff = differentiator(diff_window, dt)
half_diff_window = (diff_window - 1) // 2

# Initializing augmenting controller
aug_controller = ConstantController(output, zeros(m))

# Initializing data arrays
t_episodes = zeros(0)
x_episodes = zeros((0, n))
decoupling_episodes = zeros((0, m))
input_episodes = zeros((0, s))
u_nom_episodes = zeros((0, m))
u_pert_episodes = zeros((0, m))
V_dot_r_episodes = zeros(len(t_episodes))

# Initializing additional data arrays
V_dot_episodes = zeros(0)
V_dot_true_episodes = zeros(0)
a_episodes = zeros((0, m))
a_true_episodes = zeros((0, m))
b_episodes = zeros(0)
b_true_episodes = zeros(0)

# True output Lyapunov function object for comparison
output_true = InvertedPendulumOutput(system_true, t_ds, x_ds)
lyapunov_function_true = QuadraticControlLyapunovFunction.build_care(output_true, Q)

# Sigmoid weighting
weight_final = 0.99
weights = sigmoid_weighting(num_episodes, weight_final)

for episode, weight in enumerate(weights):
	print('EPISODE', episode)

	# Initializing and connecting models
	a = two_layer_nn(s, d_hidden, (m,))
	b = two_layer_nn(s, d_hidden, (1,))
	model = connect_models(a, b)
	model.compile('adam', 'mean_squared_error')

	# Creating baseline controller
	nom_controller = CombinedController([pd_controller, aug_controller], array([1, weight]))

	# Creating perturbations
	perturbations = uniform(-width, width, (N // subsample_rate, m))
	perturbations = reshape(tile(perturbations, [1, subsample_rate]), (-1, m))
	pert_controller = PerturbingController(output, nom_controller, t_eval, perturbations, offset=offset)

	# Simulating experiment
	total_controller = CombinedController([nom_controller, pert_controller], ones(2))
	ts, xs = system_true.simulate(x_0, total_controller, t_eval)

	# Post processing simulation data
	Vs = array([lyapunov_function.V(x, t) for x, t in zip(xs, ts)])
	xs = xs[half_diff_window:-half_diff_window:subsample_rate] # Removing data truncated by diff filter, subsampling
	ts = ts[half_diff_window:-half_diff_window:subsample_rate] # Removing data truncated by diff filter, subsampling
	inputs = array([input(x, t) for x, t in zip(xs, ts)])
	u_noms = array([nom_controller.u(x, t) for x, t in zip(xs, ts)])
	u_perts = array([pert_controller.u(x, t) for x, t in zip(xs, ts)])
	grad_Vs = array([lyapunov_function.grad_V(x, t) for x, t in zip(xs, ts)])
	decouplings = array([lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])

	# Label estimation
	V_dots = diff(Vs)[::subsample_rate]
	V_dot_ds = array([lyapunov_function.V_dot(x, u_nom, t) for x, u_nom, t in zip(xs, u_noms, ts)])
	V_dot_rs = V_dots - V_dot_ds

	# Aggregating data
	decoupling_episodes = concatenate([decoupling_episodes, decouplings])
	input_episodes = concatenate([input_episodes, inputs])
	u_nom_episodes = concatenate([u_nom_episodes, u_noms])
	u_pert_episodes = concatenate([u_pert_episodes, u_perts])
	V_dot_r_episodes = concatenate([V_dot_r_episodes, V_dot_rs])

	# Shuffling data
	perm = permutation(len(input_episodes))
	decoupling_trains = decoupling_episodes[perm]
	input_trains = input_episodes[perm]
	u_nom_trains = u_nom_episodes[perm]
	u_pert_trains = u_pert_episodes[perm]
	V_dot_r_trains = V_dot_r_episodes[perm]

	# Fitting model
	model.fit([decoupling_trains, input_trains, u_nom_trains, u_pert_trains], V_dot_r_trains, epochs=max_epochs, callbacks=[TrainingLossThreshold(loss_threshold)], batch_size=(len(input_episodes) // 10), validation_split=validation_split)

	# Additional post processing
	V_dot_trues = [lyapunov_function_true.V_dot(x, u_nom + u_pert, t) for x, u_nom, u_pert, t in zip(xs, u_noms, u_perts, ts)]
	V_dot_r_debugs = V_dot_trues - V_dot_ds
	_as = a.predict(inputs)
	bs = b.predict(inputs)[:, 0]
	a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
	b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])

	# Aggregating additional data
	V_dot_episodes = concatenate([V_dot_episodes, V_dots])
	V_dot_true_episodes = concatenate([V_dot_true_episodes, V_dot_trues])
	x_episodes = concatenate([x_episodes, xs])
	a_episodes = concatenate([a_episodes, _as])
	b_episodes = concatenate([b_episodes, bs])
	a_true_episodes = concatenate([a_true_episodes, a_trues])
	b_true_episodes = concatenate([b_true_episodes, b_trues])

	# Building new augmenting controller
	a = evaluator(input, a)
	b = evaluator(input, b, scalar_output=True)
	aug_controller = QPController.build_aug(pd_controller, m, lyapunov_function, a, b, C)

# Building final augmented controller
total_controller = CombinedController([pd_controller, aug_controller], ones(2))

# Nominal QP controller simulation
ts, xs = system_true.simulate(x_0, qp_controller, t_eval)

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('QP Controller')
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# PD controller simulation
ts, xs = system_true.simulate(x_0, pd_controller, t_eval)

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('PD Controller')
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# Augmented controller simulation
ts, xs = system_true.simulate(x_0, total_controller, t_eval)

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('Augmented Controller')
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# Additional evaluation plots

figure()
suptitle('Episode data')
subplot(2, 1, 1)
title('State data')
plot(x_episodes)
legend(['$\\theta$', '$\\dot{\\theta}$'], fontsize=16)
grid()
subplot(2, 1, 2)
title('Control data')
plot(u_nom_episodes + u_pert_episodes)
legend(['$u$'], fontsize=16)
grid()

figure()
plot(V_dot_episodes, linewidth=2)
plot(V_dot_true_episodes, '--', linewidth=2)
title('Numerical differentiation')
legend(['Estimated $\\dot{V}$', 'True $\\dot{V}$'], fontsize=16)
grid()

figure()
suptitle('Training results')
subplot(2, 1, 1)
plot(a_episodes)
plot(a_true_episodes, '--')
title('$a$')
legend(['Estimated', 'True'], fontsize=16)
grid()
subplot(2, 1, 2)
plot(b_episodes)
plot(b_true_episodes, '--')
title('$b$')
legend(['Estimated', 'True'], fontsize=16)
grid()

# Testing post processing
_as = array([a(x, t) for x, t in zip(xs, ts)])
bs = array([b(x, t) for x, t in zip(xs, ts)])
a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])

figure()
suptitle('Testing results')
subplot(2, 1, 1)
plot(_as)
plot(a_trues, '--')
title('a')
legend(['Estimated', 'True'], fontsize=16)
grid()
subplot(2, 1, 2)
plot(bs)
plot(b_trues, '--')
title('b')
legend(['Estimated', 'True'], fontsize=16)
grid()

show()
