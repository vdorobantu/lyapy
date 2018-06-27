"""Utilities for learning problems"""

from cvxpy import Minimize, Variable, Problem, square
from keras.layers import Add, Dot, Input
from keras.models import Model
from numpy import arange, array, convolve, dot, power, reshape, zeros
from numpy.linalg import inv

def connect_models(a, b):
    """Connect two regression models a, b to form h(x, u) = a(x)' * u + b(x).

    x is in R^n and u is in R^m. Outputs keras model (R^n * R^m -> R).

    Inputs:
    Regression model, a: keras Sequential model (R^n -> R^m)
    Regression model, b: keras Sequential model (R^n -> R)
    """

    n, m = a.input_shape[-1], a.output_shape[-1]
    x, u = Input((n,)), Input((m,))
    a, b = a(x), b(x)
    dV_hat = Dot(1)([a, u])
    dV_hat = Add()([b, dV_hat])
    model = Model([x, u], dV_hat)
    return model

def differentiator(L, h):
    """Create L-step causal differentiator filter.

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

def evaluator(model):
    """Convert keras model to callable function.

    Outputs function mapping numpy array (n,) * numpy array (m,) to float.

    Inputs:
    Regression model, model: keras model
    """

    def f(x):
        xs = array([x])
        ys = model.predict(xs)
        return ys[0]

    return f

def constant_controller(u_0):
    """Create controller that always outputs constant.

    Outputs function mapping numpy array (n,) * float to float.

    Inputs:
    Constant value, u_0: numpy array (n,) * float -> float
    """

    def u_const(x, t):
        return u_0

    return u_const

def sum_controller(us):
    """Create controller that outputs sum of outputs of a list of controllers.

    Outputs function mapping numpy array (n,) * float to float.

    Inputs:
    List of controllers, us: (numpy array (n,) * float -> float) list
    """

    def u_sum(x, t):
        return sum([u(x, t) for u in us])

    return u_sum

def augmenting_controller(u, a, b, dV, C):
    """Create augmenting controller to enforce Lyapunov function time derivative.

    Outputs function mapping numpy array (n,) * float to float.

    Inputs:
    Nominal controller, u: numpy array (n,) * float -> float
    Callable regression model, a: numpy array (n,) -> numpy array (m,)
    Callable regression model, b: numpy array (n,) -> float
    Desired Lyapunov function derivative, dV: numpy array (n,) * numpy array (m,) * float -> float
    Relaxation variable coefficient, C: float
    """

    a = evaluator(a)
    b = evaluator(b)

    def u_aug(x, t):
        u_c = u(x, t)
        m = len(u_c)
        u_l = Variable(m)
        eps = Variable()
        obj = Minimize(1 / 2 * (square(u_c + u_l) + C * square(eps)))
        cons = [a(x) * (u_c + u_l) + b(x) <= dV(x, u_c, t) + eps, eps >= 0]
        prob = Problem(obj, cons)
        prob.solve()
        print(t, eps.value)
        return u_l.value

    return u_aug
