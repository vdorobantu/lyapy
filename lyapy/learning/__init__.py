"""Learning utilities.

connect_models - Connect two keras models affinely.
decay_widths - Compute permuting controller widths for specified number of episodes.
differentiator - Create L-step centered differentiator filter.
evaluator - Create a function wrapped around keras model predict call.
KerasHandler - Trainer object for Keras models.
sigmoid_weighting - Compute weights governed by a sigmoid function for specified number of weights and final weight.
SimulationHandler - Handler object for native simulations.
TrainingLossThreshold - Class to stop keras training when training loss falls below a threshold.
two_layer_nn - Create a two-layer neural network.
"""

from .keras_trainer import KerasTrainer
from .simulation_handler import SimulationHandler
from .util import connect_models, decay_widths, differentiator, evaluator, sigmoid_weighting, TrainingLossThreshold, two_layer_nn
