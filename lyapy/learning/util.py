"""Utilities for learning problems"""

from keras.callbacks import Callback
from keras.layers import Add, Dense, Dot, Dropout, Input, Reshape
from keras.models import Model, Sequential
from numpy import arange, array, convolve, dot, linspace, power, product, reshape, tile, zeros
from numpy.linalg import inv, solve

from ..controllers import PDController

def sigmoid_weighting(num_episodes, weight_final):
	"""Compute weights governed by a sigmoid function for specified number of weights and final weight.

	Let T be the number of weights.

	First episode weight is 1 - (final weight). If number of weights is odd,
	weight for episode (T - 1) / 2 is 0.5.

	Outputs a numpy array (T,).

	Inputs:
	Number of episodes, num_episodes: int
	Final weight: weight_final: float
	"""

	return 1 / (1 + ((1 - weight_final) / (weight_final)) ** (2 * linspace(0, 1, num_episodes) - 1))

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

def differentiator(L, h):
	"""Create L-step centered differentiator filter.

	Outputs function mapping numpy array (N,) to numpy array (N - L + 1,).

	Inputs:
	Size of filter, L: int
	Sample time, h: float
	"""

	ks = reshape(arange(L), (L, 1))
	A = power(ks.T, ks)
	b = zeros(L)
	b[1] = -1 / h
	w = dot(inv(A), b)

	def diff(xs):
		return convolve(w, xs, 'valid')

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
