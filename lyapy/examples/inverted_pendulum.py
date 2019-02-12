"""Inverted pendulum example."""

from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import arange, array, concatenate, identity, ones, sin
from numpy.random import uniform
from scipy.io import loadmat

from ..controllers import PDController, QPController, CombinedController
from ..learning import decay_widths, evaluator, KerasTrainer, sigmoid_weighting, SimulationHandler
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
batch_fraction = 0.1
validation_split = 0.1 # Percentage of training data withheld for validation
subsample_rate = 20 # Number of time steps perturbations are held for
width = 0.1 # Width of uniform distribution of perturbations
scaling = 1
offset = 0.1 # Perturbation offset
d_hidden = 200 # Hidden layer dimension
C = 1e3 # Augmenting controller slack weight
weight_final = 0.99
add_episodes = 0

# Numerical differentiation filter
diff_window = 3

# Run experiments in simulation
handler = SimulationHandler(system_true, output, pd_controller, m, lyapunov_function, x_0, t_eval, subsample_rate, input, C, scaling, offset)
# Set up episodic learning with Keras
trainer = KerasTrainer(input, lyapunov_function, diff_window, subsample_rate, n, s, m, d_hidden, loss_threshold, max_epochs, batch_fraction, validation_split)
weights = sigmoid_weighting(num_episodes, weight_final)
widths = width * ones(num_episodes)

# Train and build augmented controller
a, b, train_data, (a_predicts, b_predicts) = trainer.run(handler, weights, widths)
a = evaluator(input, a)
b = evaluator(input, b, scalar_output=True)
aug_controller = QPController.build_aug(pd_controller, m, lyapunov_function, a, b, C)
total_controller = CombinedController([pd_controller, aug_controller], ones(2))

# True output Lyapunov function object for comparison
output_true = InvertedPendulumOutput(system_true, t_ds, x_ds)
lyapunov_function_true = QuadraticControlLyapunovFunction.build_care(output_true, Q)

# Nominal QP controller simulation
ts, xs = system_true.simulate(x_0, qp_controller, t_eval)

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('QP Controller', fontsize=16)
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# PD controller simulation
ts, xs = system_true.simulate(x_0, pd_controller, t_eval)

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('PD Controller', fontsize=16)
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# Augmented controller simulation
ts, xs = system_true.simulate(x_0, total_controller, t_eval)

figure()
plot(ts, xs, linewidth=2)
plot(t_ds, x_ds, '--', linewidth=2)
title('Augmented Controller', fontsize=16)
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=16)
grid()

# Additional evaluation plots
a_tests = array([a(x, t) for x, t in zip(xs, ts)])
a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
b_tests = array([b(x, t) for x, t in zip(xs, ts)])
b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])

figure()
suptitle('Test data', fontsize=16)
subplot(2, 1, 1)
plot(a_tests, label='$\\widehat{a}$', linewidth=2)
plot(a_trues, label='$a$', linewidth=2)
legend(fontsize=16)
grid()
subplot(2, 1, 2)
plot(b_tests, label='$\\widehat{b}$', linewidth=2)
plot(b_trues, label='$b$', linewidth=2)
legend(fontsize=16)
grid()

xs, ts, _, _, u_noms, u_perts, V_dot_rs = train_data
a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])
V_dot_r_trues = array([lyapunov_function_true.V_dot(x, u_nom + u_pert, t) - lyapunov_function.V_dot(x, u_nom, t) for x, u_nom, u_pert, t in zip(xs, u_noms, u_perts, ts)])

figure()
suptitle('Episode data', fontsize=16)
subplot(2, 1, 1)
title('State data', fontsize=16)
plot(xs, linewidth=2)
legend(['$\\theta$', '$\\dot{\\theta}$'], fontsize=16)
grid()
subplot(2, 1, 2)
title('Control data', fontsize=16)
plot(u_noms + u_perts, label='$u$', linewidth=2)
legend(fontsize=16)
grid()

figure()
plot(V_dot_rs, linewidth=2, label='$\\widehat{\\dot{V}}_r$')
plot(V_dot_r_trues, '--', linewidth=2, label='$\\dot{V}_r$')
title('Numerical differentiation', fontsize=16)
legend(fontsize=16)
grid()

figure()
suptitle('Training results', fontsize=16)
subplot(2, 1, 1)
plot(a_predicts, label='$\\widehat{a}$', linewidth=2)
plot(a_trues, '--', label='$a$', linewidth=2)
legend(fontsize=16)
grid()
subplot(2, 1, 2)
plot(b_predicts, label='$\\widehat{b}$', linewidth=2)
plot(b_trues, '--', label='$b$', linewidth=2)
legend(fontsize=16)
grid()

show()
