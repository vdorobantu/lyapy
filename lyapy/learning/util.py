from keras.layers import Add, Dot, Input
from keras.models import Model
from numpy import arange, convolve, dot, power, reshape, zeros
from numpy.linalg import inv

def connect_models(w, b):
    n, m = w.input_shape[-1], w.output_shape[-1]
    x, u = Input((n,)), Input((m,))
    w, b = w(x), b(x)
    dV_hat = Dot(1)([w, u])
    dV_hat = Add()([b, dV_hat])
    model = Model([x, u], dV_hat)
    return model

def differentiator(L, h):
    ks = reshape(arange(L), (L, 1))
    A = power(ks.T, ks)
    b = zeros(L)
    b[1] = -1 / h
    w = dot(inv(A), b)

    def diff(xs):
        return convolve(w, xs, 'valid')

    return diff
