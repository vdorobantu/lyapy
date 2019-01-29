"""Learning utilities.

connect_models - Connect two keras models affinely.
differentiator - Create L-step centered differentiator filter.
evaluator - Create a function wrapped around keras model predict call.
sigmoid_weighting - Compute weights governed by a sigmoid function for specified number of weights and final weight.
TrainingLossThreshold - Class to stop keras training when training loss falls below a threshold.
two_layer_nn - Create a two-layer neural network.
"""

from .util import connect_models, differentiator, evaluator, sigmoid_weighting, TrainingLossThreshold, two_layer_nn
