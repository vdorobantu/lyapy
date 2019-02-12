"""Utilities for learning problems"""

from keras.callbacks import Callback
from keras.layers import Add, Dense, Dot, Dropout, Input, Reshape
from keras.models import Model, Sequential
from numpy import arange, array, concatenate, convolve, dot, linspace, ones, power, product, reshape, tile, zeros
from numpy.linalg import inv, solve

from ..controllers import PDController

def sigmoid_weighting(num_episodes, weight_final, add_episodes=0):
	"""Compute weights governed by a sigmoid function for specified number of weights and final weight.

	Let T be the number of weights, excluding additional start and end weights.

	First episode weight is 1 - (final weight). If number of weights is odd,
	weight for episode (T - 1) / 2 is 0.5.

	Outputs a numpy array (T + 2 * add_episodes,).

	Inputs:
	Number of episodes, num_episodes: int
	Final weight: weight_final: float
	Number of start weights (or end weights), add_episodes: int
	"""

	weights = 1 / (1 + ((1 - weight_final) / (weight_final)) ** (2 * linspace(0, 1, num_episodes) - 1))
	return concatenate([zeros(add_episodes), weights, ones(add_episodes)])

def decay_widths(num_episodes, width, add_episodes):
	"""Compute permuting controller widths for specified number of episodes.

	Let T be the number of episodes, excluding additional episodes.

	First T + add_episodes episodes use nominal width, then width is reduced
	linearly for last add_episodes episodes.

	Outputs a numpy array (T + 2 * add_episodes,).

	Inputs:
	Number of episodes, num_episodes: int
	Nominal width, width: float
	Number of decaying width episodes, add_episodes: int
	"""

	return concatenate([width * ones(add_episodes + num_episodes), linspace(width, 0, add_episodes)])

def two_layer_nn(d_in, d_hidden, output_shape, dropout_prob=0):
	"""Create a two-layer neural network.

	Uses Rectified Linear Unit (ReLU) nonlinearity.

	Outputs keras model (R^d_in -> R^output_shape).

	Inputs:
	Input dimension, d_in: int
	Hidden layer dimension, d_hidden: int
	Output shape, output_shape: int tuple
	Dropout regularization probability, dropout_prob: float
	"""

	model = Sequential()
	model.add(Dense(d_hidden, input_shape=(d_in,), activation='relu'))
	model.add(Dropout(dropout_prob))
	model.add(Dense(product(output_shape)))
	model.add(Reshape(output_shape))
	return model

def connect_models(a, b):
	"""Connect two keras models affinely.

	The models, a and b, are connected as decoupling * u_nom + a(input)' u_pert
	+ b(input).

	Let s be the input size of both models, m be the number of inputs.

	Outputs keras model (R^m * R^s * R^m * R^m -> R).

	Inputs:
	Linear term, a: keras model (R^s -> R^m)
	Scalar term, b: keras model (R^s -> R)
	"""

	s, m = a.input_shape[-1], a.output_shape[-1]

	input = Input((s,))
	u_nom = Input((m,))
	u_pert = Input((m,))

	decoupling = Input((m,))

	a = a(input)
	b = b(input)

	u = Add()([u_nom, u_pert])
	known = Dot([1, 1])([decoupling, u_pert])
	unknown = Add()([Dot([1, 1])([a, u]), b])

	V_dot_r = Add()([known, unknown])

	return Model([decoupling, input, u_nom, u_pert], V_dot_r)

class TrainingLossThreshold(Callback):
	"""Class to stop keras training when training loss falls below a threshold.

	Attributes:
	Loss threshold, loss_threshold: float
	"""

	def __init__(self, loss_threshold):
		Callback.__init__(self)
		self.loss_threshold = loss_threshold

	def on_epoch_end(self, epoch, logs):
		"""Stops training when training loss falls below loss_threshold.

		Inputs:
		Epoch number, epoch: int
		Keras log object, logs: Keras log
		"""

		if logs.get('loss') < self.loss_threshold:
			self.model.stop_training = True

def differentiator(L):
	half_L = (L - 1) // 2
	ks = arange(L)
	b = array([0, 1] + ([0] * (L - 2)))

	def _diff(xs, ts):
		ts = ts - ts[half_L]
		A = (array([ts]).T ** ks).T
		w = solve(A, b)
		return dot(w, xs)

	def diff(xs, ts):
		N = len(xs)
		return array([_diff(xs[i:(i + L)], ts[i:(i + L)]) for i in range(N - L + 1)])

	return diff

def evaluator(input, model, scalar_output=False):
	"""Create a function wrapped around keras model predict call.

	Let n be the number of states, m be the number of inputs, s be the input
	size of the model.

	Outputs a numpy array (n,) * float -> {numpy array (m,), float}.

	Inputs:
	Input function, input: numpy array (n,) * float -> numpy array (s,)
	Keras model, model: keras model (R^s -> {R^m, R})
	Flag indicating if output is scalar, scalar_output: bool
	"""

	def f(x, t):
		inputs = array([input(x, t)])
		ys = model.predict(inputs)
		y = ys[0]
		if scalar_output:
			y = y[0]
		return y

	return f
