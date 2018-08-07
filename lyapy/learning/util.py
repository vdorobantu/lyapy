"""Utilities for learning problems"""

from keras.layers import Add, Dense, Dot, Dropout, Input, Reshape
from keras.models import Model, Sequential
from numpy import arange, array, concatenate, convolve, dot, power, product, reshape, zeros
from numpy.linalg import inv, norm
from numpy.random import randn
from random import sample
from keras.callbacks import EarlyStopping


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

def runEpisodes(u_c, u_l_prev, u_l_epsilons_prev, a_model, b_model, initialConditions, otherConds, sysConds, existing_xdata = [], existing_tdata = [], existing_solsdata = []):
    """ Learn an augmented controller for a system based on data and/or a learning model from a previous episode

    Outputs:
    Updated regression model, a_model: keras Sequential model (R^n -> R^(n * m))
    Updated regression model, b_model: keras Sequential model (R^n -> R^n)
    Recently learned controller, u_l: numpy array (n,) * float -> float
    Updated learned controller + perturbations, u_l_epsilons_prev: numpy array (n * num_episodes,)
    Recently generated state space data, xs: numpy array  (N, n)
    Recently generated time data, ts:  numpy array (N,)
    Recently generated times and corresponding solutions, sols: numpy array (N,) * numpy array (N, n)

    Inputs:
    Nominal controller, u_c: numpy array (n,) * float -> float
    Learned controller, u_l_prev: numpy array (n,) * float -> float
    Learned controller + perturbations, u_l_epsilons_prev: numpy array (n * num_episodes,)
    Regression model, a_model: keras Sequential model (R^n -> R^(n * m))
    Regression model, b_model: keras Sequential model (R^n -> R^n)
    Function to generate initial conditions, initialConditions: numpy array (n,) -> numpy array (n,)
    Learning-related conditions, otherConds: n-tuple of floats 
    System-related conditions, sysConds: n-tuple of system objects and functions
    State space data from previous episodes, existing_xdata: numpy array  (N, n)
    Time data from previous episodes, existing_tdata: numpy array (N,)
    Times and corresponding solutions from previous episodes, existing_solsdata: numpy array (N,) * numpy array (N, n)

    """

    # Learning-related conditions, otherConds, tuple unpacked
    #   i:  number
    #   numdiff: length of differentiation filter
    #   numind: number of points per trajectory to be trained on
    #   N: number of trajectories to generate per episode
    #   sigma: variance of perturbation
    #   dt: sample time
    #   num_timesteps: number of timesteps to simulate training data
    #   n_epochs: maximum number of epochs to run model
    i, numdiff, numind, N, sigma, dt, num_timesteps, n_epochs = otherConds

    # System-related conditions, sysConds, tuple unpacked
    #   system: system object
    #   V: Lyapunov function of system
    #   dVdx: Lyapunov function gradient of system
    #   dV: Lyapunov function time derivative of system
    system, V, dVdx, dV = sysConds
    
    t_evals, x_0s = initialConditions(N, dt, num_timesteps) # generate initial conditions to run simulations
                                                            # to create training data
    
    diff = differentiator(numdiff, dt) # Differentiator filter

    epsilons = [constant_controller(u_0) for u_0 in sigma * randn(N, 1)] # Constant perturbations
    u_augs = [sum_controller([u_c, u_l_prev, epsilon]) for epsilon in epsilons] # Controller + constant perturbations

    model = connect_models(a_model, b_model) # create model that will output final learned controller
    model.compile('adam', 'mean_squared_error')

    dataset = [system.simulate(u_aug, x_0, t_eval) for u_aug, x_0, t_eval in zip(u_augs, x_0s, t_evals)] # generate training dataset
        
    # get numind solutions, xs, ts, and perturbations per trajectory 
    tempxs = [] 
    tempts = []
    sols = []
    totaleps = []
    
    for idx, (ts, xs) in enumerate(dataset):
        rand_indexes = sample(range(numdiff - 1, num_timesteps), numind) # sample without replacement
        for rand_index in rand_indexes:
            totaleps.append(epsilons[idx])
            tempxs.append(xs[rand_index])
            tempts.append(ts[rand_index])
            sols_ts = []
            sols_xs = []
            for k in list(range(numdiff - 1, -1, -1)):
                sols_ts.append(ts[rand_index - k])
                sols_xs.append(xs[rand_index - k])
            sols.append((array(sols_ts), array(sols_xs)))

    # append recently generated training data to existing training data if it exists
    if (existing_xdata != [] and existing_tdata != []):
        xs = concatenate((existing_xdata, array(tempxs)))
        ts = concatenate((existing_tdata, array(tempts)))
        existing_solsdata.extend(sols)
        sols = existing_solsdata
    else:
        xs = array(tempxs)
        ts = array(tempts)

    u_l_epsilons = [sum_controller([u_l_prev, totalep]) for totalep in totaleps] # Controller + constant perturbations
    u_l_epsilons_prev.extend(u_l_epsilons) # add controller + constant perturbations for recently generated data
                                           # to controller + constant perturbation used to train already existing data
    
    u_ls = array([u_l_epsilon(x, t) for x, t, u_l_epsilon in zip(xs, ts, u_l_epsilons_prev)])
    u_cs = array([u_c(x, t) for x, t in zip(xs, ts)])
    
    # Numerically differentiating V for each simulation
    dV_hats = concatenate([diff(array([V(x, t) for x, t in zip(xs, ts)])) for ts, xs in sols])
    dV_ds = array([dV(x, u_c, t) for x, u_c, t in zip(xs, u_cs, ts)])
    dV_r_hats = dV_hats - dV_ds

    dVdxs = array([dVdx(x, t) for x, t in zip(xs, ts)])
    gs = array([system.act(x) for x in xs])

    # add early stopping, another regularization parameter
    earlystop = EarlyStopping(monitor='val_loss', min_delta=10, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    model.fit([dVdxs, gs, xs, u_cs, u_ls], dV_r_hats, epochs=n_epochs, callbacks=callbacks_list, batch_size=len(xs)//10, validation_split=0.5)

    # save recently trained model and its components
    model.save('model.h5')
    a_model.save('a.h5') 
    b_model.save('b.h5')
        
    C = 1e3
    u_l = augmenting_controller(dVdx, system.act, u_c, a_model, b_model, C)
    
    return a_model, b_model, u_l, u_l_epsilons_prev, xs, ts, sols





    


