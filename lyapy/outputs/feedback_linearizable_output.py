"""Base class for feedback linearizable outputs."""

from numpy import arange, argsort, concatenate, cumsum, delete, diag, dot, ones, zeros
from scipy.linalg import block_diag

from .affine_dynamic_output import AffineDynamicOutput

class FeedbackLinearizableOutput(AffineDynamicOutput):
    """Base class for feedback linearizable outputs.

    Override eta, drift, decoupling.

    Let n be the number of states, k be the number of outputs, p be the output
    vector size.

    For outputs with relative degrees gamma_1, ..., gamma_k, output vector is
    block vector with i-th block containing output and corresponding derivatives
    up to degree (gamma_i - 1).

    Output dynamics are eta_dot(x, t) = drift(x, t) + decoupling(x, t) * u.

    If output vector is not in block form, must provide indices of permuation
    into block form. Indices are specified as (i_1, ..., i_p) and transform
    (eta_1, ..., eta_p) into (eta_(i_1), ... eta_(i_p)).

    Attributes:
    List of relative degrees, vector_relative_degree: int list
    Permutation indices, permutation_idxs: numpy array (p,)
    Reverse permutation indices, reverse_permutation_idxs: numpy array (p,)
    Indices of k outputs when eta in block form, relative_degree_idxs: numpy array (k,)
    Indices of permutation into form with highest order derivatives in block, blocking_idxs: numpy array (p,)
    Indices of reverse permutation into form with highest order derivatives in block, unblocking_idxs: numpy array (p,)
    Linear output update matrix after decoupling inversion and drift removal, F: numpy array (p, p)
    Linear output actuation matrix after decoupling inversion and drift removal, G: numpy array (p, k)
    """

    def __init__(self, vector_relative_degree, permutation_idxs=None):
        """Initialize a FeedbackLinearizableOutput object.

        Inputs:
        List of relative degrees, vector_relative_degree: int list
        Permutation indices, permutation_idxs: numpy array (p,)
        """

        self.vector_relative_degree = vector_relative_degree
        output_size = sum(vector_relative_degree)
        if permutation_idxs is None:
            permutation_idxs = arange(output_size)
        self.permutation_idxs = permutation_idxs
        self.reverse_permutation_idxs = argsort(permutation_idxs)
        self.relative_degree_idxs = cumsum(vector_relative_degree) - 1
        non_relative_degree_idxs = delete(arange(output_size), self.relative_degree_idxs)
        self.blocking_idxs = concatenate([non_relative_degree_idxs, self.relative_degree_idxs])
        self.unblocking_idxs = argsort(self.blocking_idxs)

        F = block_diag(*[diag(ones(gamma - 1), 1) for gamma in vector_relative_degree])
        G = block_diag(*[concatenate([zeros(gamma - 1), ones(1)]) for gamma in vector_relative_degree]).T

        self.F = self.reverse_permute(self.reverse_permute(F).T).T
        self.G = self.reverse_permute(G)

    def permute(self, arr):
        """Apply permuation to array.

        Outputs a numpy array (p, ...).

        Inputs:
        Array, arr: numpy array (p, ...)
        """

        return arr[self.permutation_idxs]

    def reverse_permute(self, arr):
        """Apply reversed permuation to array.

        Outputs a numpy array (p, ...).

        Inputs:
        Array, arr: numpy array (p, ...)
        """

        return arr[self.reverse_permutation_idxs]

    def block(self, arr):
        """Apply permuation to array (into form with highest order derivatives in block), relative to result of initial permuation.

        Outputs a numpy array (p, ...).

        Inputs:
        Array, arr: numpy array (p, ...)
        """

        return arr[self.blocking_idxs]

    def unblock(self, arr):
        """Apply reverse permuation to array (into form with highest order derivatives in block), relative to result of initial permutation.

        Outputs a numpy array (p, ...).

        Inputs:
        Array, arr: numpy array (p, ...)
        """

        return arr[self.unblocking_idxs]

    def select(self, arr):
        """Select elements of array corresponding to highest order derivative terms, relative to result of initial permuation.

        Outputs a numpy array (p, ...).

        Inputs:
        Array, arr: numpy array (p, ...)
        """

        return arr[self.relative_degree_idxs]

    def closed_loop_dynamics(self, K):
        """Computes the linear closed loop dynamics matrix of a feedback linearizable output controlled with a linearizing  feedback controller with a specified auxilliary control gain matrix.

        Outputs a numpy array (p, p).

        Inputs:
        Auxilliary control gain matrix, K: numpy array (k, p)
        """

        return self.F - dot(self.G, K)
