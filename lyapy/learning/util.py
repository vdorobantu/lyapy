"""Utilities for learning problems"""

from keras.layers import Add, Dense, Dot, Dropout, Input, Reshape
from keras.models import Model, Sequential
from numpy import arange, array, convolve, dot, power, product, reshape, zeros
from numpy.linalg import inv, norm

def two_layer_nn(d_in, d_hidden, output_shape, dropout_prob=0):
    """Create a two-layer neural network.

    Uses Rectified Linear Unit (ReLU) nonlinearity. Outputs keras model
    (R^d_in -> R^output_shape).

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
    """Connect two regression models a, b to form h(x, u) = dVdx * (g(x) * u_l + a(x) * (u_c + u_l) + b(x)).

    x is in R^n and u_c, u_l are in R^m. Outputs keras model
    (R^n * R^(n * m) * R^n * R^m * R^m -> R).

    Inputs:
    Regression model, a: keras Sequential model (R^n -> R^(n * m))
    Regression model, b: keras Sequential model (R^n -> R^n)
    """

    n, m = a.output_shape[1:3]
    x, u_c, u_l = Input((n,)), Input((m,)), Input((m,))
    dVdx, g = Input((n,)), Input((n, m))
    a, b = a(x), b(x)
    gu_l = Dot([2, 1])([g, u_l])
    u = Add()([u_c, u_l])
    au = Dot([2, 1])([a, u])
    sum = Add()([gu_l, au, b])
    V_dot_r = Dot([1, 1])([dVdx, sum])
    model = Model([dVdx, g, x, u_c, u_l], V_dot_r)
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
    Constant value, u_0: numpy array (n,)
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

def augmenting_controller(dVdx, g, u, a, b, C):
    """Create augmenting controller to enforce Lyapunov function time derivative.

    Outputs function mapping numpy array (n,) * float to float.

    Inputs:
    Callable Lyapunov gradient, dVdx: numpy array (n,) * float -> numpy array (n,)
    Callable actuation matrix, g: numpy array (n,) -> numpy array (n, m)
    Nominal controller, u: numpy array (n,) * float -> float
    Callable regression model, a: numpy array (n,) -> numpy array (n, m)
    Callable regression model, b: numpy array (n,) -> numpy array (n,)
    Relaxation variable coefficient, C: float
    """

    a = evaluator(a)
    b = evaluator(b)

    def u_aug(x, t):
        lambda_r = dot(dVdx(x, t), -dot(g(x), u(x, t)) + b(x)) / (norm(dot(dVdx(x, t), (g(x) + a(x)))) ** 2 + 1 / C)
        lambda_r = max(0, lambda_r)
        lambda_plus = 0
        u_l = -u(x, t) - lambda_r * dot(dVdx(x, t), g(x) + a(x))
        epsilon = (lambda_r + lambda_plus) / C
        print(t, epsilon)
        return u_l

    return u_aug
