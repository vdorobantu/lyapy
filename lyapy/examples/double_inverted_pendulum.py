"""Double inverted pendulum example."""

from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import arange, array, concatenate, identity, ones, sin, cos
from numpy.random import uniform
from scipy.io import loadmat

from ..controllers import PDController, QPController, CombinedController
from ..learning import evaluator, KerasTrainer, SimulationHandler
from ..lyapunov_functions import QuadraticControlLyapunovFunction
from ..outputs import RoboticSystemOutput
from ..systems import AffineControlSystem

class DoubleInvertedPendulum(AffineControlSystem):
	"""Double Inverted pendulum model.

	States are x = (theta1, theta1_dot, theta2, theta2_dot),


    where theta is the angle of the pendulum
	in rad clockwise from upright and theta_dot is the angular rate of the
	pendulum in rad/s clockwise. The input is u = (tau), where tau is torque in
	N * m, applied clockwise at the base of the pendulum.

	Attributes:
	Mass (kg), m: float
	Gravitational acceleration (m/s^2), g: float
	Length (m), l: float
	"""

	def __init__(self, m1, m2, g, l1, l2):
		"""Initialize an DoubleInvertedPendulum object.

		Inputs:
		Mass of lower part (kg), m1: float
		Mass of upper part (kg), m2: float
		Gravitational acceleration (m/s^2), g: float
		Length of lower part (m), l1: float
		Length of upper part (m), l2: float
		"""

		AffineControlSystem.__init__(self)
		self.m1, self.m2 self.g, self.l1, self.l2 = m1, m2, g, l1, l2

	def inv_matrix(a,b,c,d):
		det = ad - bc
		# handle the zero case?
		return [[d / det, -b / det], [-c / det, a / det]]

	def J_inv(theta1, theta2, theta1_dot, theta2_dot):
		a = (self.m1 + self.m2) * (self.l1 ** 2) + (self.m2 * (self.l2 ** 2)) + (2 * self.m2 * self.l1 * self.l2 * cos(theta2))
		b = (self.m2 * (self.l2 ** 2)) + (self.m2 * self.l1 * self.l2 * cos(theta2))
		c = b
		d = self.m2 * (self.l2 ** 2)

		return np.matrix(inv_matrix(a,b,c,d))

	def G(theta1, theta2, theta1_dot, theta2_dot):
		g = 9.8
		a = (self.m2 * self.l1 * self.l2 * (2 * theta1_dot + theta2_dot) * theta2_dot * sin(theta2)) + \
			((self.m1 + self.m2) * g * self.l1 * sin(theta1)) + (self.m2 * g * self.l1 * sin(theta1 + theta2))

		b = -(self.m2 * self.l1 * self.l2 * (theta1_dot ** 2) * sin(theta2)) + \
			(self.m2 * g * self.l2 * sin(theta1 + theta2))

		return np.matrix([a, b])

	def drift(self, x):
		theta1, theta2, theta1_dot, theta2_dot = x
		G_matrix = G(theta1, theta2, theta1_dot, theta2_dot)
		J_inv_matrix = J_inv(theta1, theta2, theta1_dot, theta2_dot)
		prod = np.matmul(G_matrix, J_inv_matrix)
		list_prod = prod.tolist()
		return array([theta1_dot, theta2_dot, list_prod[0], list_prod[1]])
		# return array([theta_dot, self.g / self.l * sin(theta)])
        # Use the equations of motion to get this drift, which is some expression computed in mathematica for J inverse

	def act(self, x):
		# Compute this J inverse in the constructor itself?
		J_inv_matrix = J_inv(theta1, theta2, theta1_dot, theta2_dot).tolist()
		return array([[0, 0], [0, 0], J_inv_matrix[0], J_inv_matrix[1]])
		# return array([[0], [1 / (self.m * (self.l ** 2))]])

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

	def __init__(self, double_inverted_pendulum, ts, x_ds):
		"""Initialize an DoubleInvertedPendulumOutput object.

		Let T be the number of time points specified.

		Inputs:
		Double inverted pendulum system, double_inverted_pendulum: DoubleInvertedPendulum
		Time points, ts: numpy array (T,)
		Desired state points, x_ds: numpy array (T, n)
		"""

		RoboticSystemOutput.__init__(self, 1)
		self.double_inverted_pendulum = double_inverted_pendulum
		self.r, self.r_dot = self.interpolator(ts, x_ds[:, :1], x_ds[:, 1:])

	def eta(self, x, t):
		# Take eta to be x
		return x

	def drift(self, x, t):
		return self.double_inverted_pendulum.drift(x) - self.r_dot(t)

	def decoupling(self, x, t):
		return self.double_inverted_pendulum.act(x)

# Parameters are unchanged currently from inverted pendulum

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
