"""Utilities for learning problems"""

from keras.callbacks import EarlyStopping
from keras.layers import Add, Dense, Dot, Dropout, Input, Reshape
from keras.models import Model, Sequential
from numpy import arange, array, concatenate, convolve, dot, pi, power, product, reshape, sin, zeros
from numpy.linalg import inv, norm
from numpy.random import uniform
from random import sample

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

    Outputs function mapping numpy array (n,) * float to numpy array (m,).

    Inputs:
    Constant value, u_0: numpy array (n,)
    """

    def u_const(x, t):
        return u_0

    return u_const

def random_controller(A, num_sinusoids, lowerfb, upperfb, u):
    """ Creates controller that outputs random perturbation proportional to the passed in controller.

    Outputs function mapping numpy array (n,) * float to numpy array (m,).

    Inputs:
    Amplitude of sinusoids, A: float
    Number of sinusoids added together to create perturbation, num_sinusoids: int
    Lower bound on freqency of sinusoids, lowerfb: float
    Upper bound on freqency of sinusoids, upperfb: float
    Controller, u: numpy array (n,) * float -> numpy array (m,).
    """
    omegas = uniform(lowerfb * 2 * pi, upperfb * 2 * pi, num_sinusoids) 
    phis = uniform(0, 2*pi, num_sinusoids) 

    def u_rand(x, t):
        pert =  sum([A * sin(omega * t + phi) for omega, phi in zip(omegas, phis)])
        return u(x, t) * pert

    return u_rand

def sum_controller(us):
    """Create controller that outputs sum of outputs of a list of controllers.

    Outputs function mapping numpy array (n,) * float to numpy array (m,).

    Inputs:
    List of controllers, us: (numpy array (n,) * float -> numpy array (m,)) list
    """

    def u_sum(x, t):
        return sum([u(x, t) for u in us])

    return u_sum

def augmenting_controller(dVdx, g, u, a, b, C):
    """Create augmenting controller to enforce Lyapunov function time derivative.

    Outputs function mapping numpy array (n,) * float to numpy array (m,).

    Inputs:
    Callable Lyapunov gradient, dVdx: numpy array (n,) * float -> numpy array (n,)
    Callable actuation matrix, g: numpy array (n,) -> numpy array (n, m)
    Nominal controller, u: numpy array (n,) * float -> numpy array (m,)
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


def generateRunEp(u_c, initialConditions, system, controller, numstates, numdiff, numind, N, sigma, dt, num_timesteps, n_epochs, randPbConds):
    """
    Creates a function that learns an augmented controller for a system based on data and/or a learning model from previous episodes

    Outputs function mapping
        Previous episodes' learned controller, u_l_prev: numpy array (n,) * float -> numpy array (m,)
        Regression model, a_model: keras Sequential model (R^n -> R^(n * m))
        Regression model, b_model: keras Sequential model (R^n -> R^n)
        Learned controllers from previous episodes, u_ls_prev: (numpy array (n, ) * float -> numpy array (m,)) list
        Perturbations from previous episodes, epsilons_prev: ((numpy array (n,) * float -> numpy array (m,)) list) list
        State space data from previous episodes, existing_xdata: numpy array  (N * (i - 1), n)
        Time data from previous episodes, existing_tdata: numpy array (N * (i - 1),)
        Times and corresponding solutions from previous episodes, existing_solsdata: numpy array (N * (i - 1),) * numpy array (N * (i - 1), n)
    to
        Updated regression model, a_model: keras Sequential model (R^n -> R^(n * m))
        Updated regression model, b_model: keras Sequential model (R^n -> R^n)
        Newly learned controller, u_l: numpy array (n,) * float -> numpy array (m,)
        Updated learned controllers from previous episodes, u_ls_prev: (numpy array (n, ) * float -> numpy array (m,)) list
        Updated perturbations from previous episodes, epsilons_prev: ((numpy array (n, ) * float -> numpy array (m,)) list) list
        Updated state space data from previous episodes, xs: numpy array  (N * i, n)
        Updated time data from previous episodes, ts:  numpy array (N * i,)
        Updated times and corresponding solutions from previous episodes, sols: numpy array (N * i,) * numpy array (N * i, n)

    
    Inputs:
    Nominal controller, u_c: numpy array (n,) * float -> numpy array (m,)
    Function to generate initial conditions, initialConditions: int * float * int -> numpy array (N,) * numpy array (N, n)
    System object, system: System
    Controller object, controller: Controller
    Number of states, numstates: int
    Length of differentiation filter, numdiff: int
    Number of points per trajectory to be trained on, numind: int
    Number of trajectories to generate per episode, N: int
    Variance of perturbation, sigma: float
    Sample time, dt: float
    Number of timesteps to simulate training data, num_timesteps: int
    Maximum number of epochs to run model, n_epochs: int
    Conditions specified when creating the random perturbations, randPbConds: float tuple
    """
    
    A, num_sinusoids, lowerfb, upperfb = randPbConds
    V, dVdx, dV = controller.V, controller.dVdx, controller.dV

    def runEpisodes(u_l_prev, a_model, b_model, u_ls_prev = [], epsilons_prev = [], existing_xdata = array([]).reshape(0, numstates), existing_tdata = array([]), existing_solsdata = []):
        t_evals, x_0s = initialConditions(N, dt, num_timesteps) # generate initial conditions to run simulations
                                                                # to create training data
        diff = differentiator(numdiff, dt) # Differentiator filter

        u_c_l = sum_controller([u_c, u_l_prev])
        epsilons = [random_controller(A, num_sinusoids, lowerfb, upperfb, u_c_l) for _ in range(N)] # generate random perturbations
        u_augs = [sum_controller([u_c, u_l_prev, epsilon]) for epsilon in epsilons] # Controller + random perturbations

        model = connect_models(a_model, b_model) # create model that will output final learned controller
        model.compile('adam', 'mean_squared_error')

        dataset = [system.simulate(u_aug, x_0, t_eval) for u_aug, x_0, t_eval in zip(u_augs, x_0s, t_evals)] # generate training dataset
            
        # get numind solutions, xs, ts, and perturbations per trajectory 
        rand_indexes = [sample(range(numdiff - 1, num_timesteps), numind) for _ in range(N)]
        newxs = [xs[rand_index] for idx, (_, xs) in enumerate(dataset) for rand_index in rand_indexes[idx] ]
        newts = [ts[rand_index] for idx, (ts, _) in enumerate(dataset) for rand_index in rand_indexes[idx] ]
        neweps = [epsilons[idx] for idx, _ in enumerate(dataset) for _ in rand_indexes[idx] ]
        sols = [(ts[rand_index - numdiff + 1:rand_index + 1], xs[rand_index - numdiff + 1:rand_index + 1]) for idx, (ts, xs) in enumerate(dataset) for rand_index in rand_indexes[idx] ]

        # append recently generated training data to existing training data if it exists
        xs = concatenate((existing_xdata, array(newxs)))
        ts = concatenate((existing_tdata, array(newts)))
        existing_solsdata.extend(sols)
        sols = existing_solsdata
            
        # add learned controllers to corresponding perturbation
        u_ls_epsilons = [sum_controller([u_l_prev, newep]) for newep in neweps] 
        u_ls_epsilons_prev = [sum_controller([u_l_past, epsilon_prev]) for u_l_past, _epsilon_prev in zip(u_ls_prev, epsilons_prev) for epsilon_prev in _epsilon_prev]
        u_ls_epsilons_prev.extend(u_ls_epsilons)

        # update list of all learned controllers and perturbations to return
        u_ls_prev.append(u_l_prev)
        epsilons_prev.append(neweps)

        # generate all data needed to train neural network
        u_ls = array([u_l_epsilon(x, t) for x, t, u_l_epsilon in zip(xs, ts, u_ls_epsilons_prev)])
        u_cs = array([u_c(x, t) for x, t in zip(xs, ts)])
        
        # Numerically differentiating V for each simulation
        dV_hats = concatenate([diff(array([V(x, t) for x, t in zip(xs, ts)])) for ts, xs in sols])
        dV_ds = array([dV(x, u_c, t) for x, u_c, t in zip(xs, u_cs, ts)])
        dV_r_hats = dV_hats - dV_ds
        
        dVdxs = array([dVdx(x, t) for x, t in zip(xs, ts)])
        gs = array([system.act(x) for x in xs])

        
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]
    
        model.fit([dVdxs, gs, xs, u_cs, u_ls], dV_r_hats, epochs=n_epochs, callbacks=callbacks_list, batch_size=len(xs)//10, validation_split=0.5)

        # save recently trained model and its components
        model.save('model.h5')
        a_model.save('a.h5') 
        b_model.save('b.h5')
            
        C = 1e3
        u_l = augmenting_controller(dVdx, system.act, u_c, a_model, b_model, C)
        
        return a_model, b_model, u_l, u_ls_prev, epsilons_prev, xs, ts, sols
    
    return runEpisodes





    


