from keras.layers import Add, Dot, Input
from keras.models import Model

def connect_models(w, b):
    n, m = w.input_shape[-1], w.output_shape[-1]
    x, u = Input((n,)), Input((m,))
    w, b = w(x), b(x)
    dV_hat = Dot(1)([w, u])
    dV_hat = Add()([b, dV_hat])
    model = Model([x, u], dV_hat)
    return model
