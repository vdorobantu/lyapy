from cvxpy import Minimize, Variable, Problem, square
from keras.layers import Add, Dot, Input
from keras.models import Model
from numpy import arange, array, convolve, dot, power, reshape, zeros
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

def evaluator(model):

    def f(x):
        xs = array([x])
        ys = model.predict(xs)
        return ys[0]

    return f

def constant_controller(u_0):

    def u_const(x, t):
        return u_0

    return u_const

def sum_controller(us):

    def u_sum(x, t):
        return sum([u(x, t) for u in us])

    return u_sum

def augmenting_controller(u, w, b, dV, C):
    w = evaluator(w)
    b = evaluator(b)

    def u_aug(x, t):
        u_c = u(x, t)
        m = len(u_c)
        u_l = Variable(m)
        eps = Variable()
        obj = Minimize(1 / 2 * (square(u_c + u_l) + C * square(eps)))
        cons = [w(x) * (u_c + u_l) + b(x) <= dV(x, u_c, t) + eps, eps >= 0]
        prob = Problem(obj, cons)
        prob.solve()
        print(t, eps.value)
        return u_l.value

    return u_aug
