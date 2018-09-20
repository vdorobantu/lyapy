"""Utilities for learning problems"""

from cvxpy import Minimize, Problem, Variable
from cvxpy import norm as cvx_norm
from keras.initializers import Constant
from keras.callbacks import EarlyStopping
from keras.layers import Add, Concatenate, Dense, Dot, Dropout, Input, Reshape
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.regularizers import l1
from numpy import arange, array, concatenate, convolve, correlate, dot, linspace, pi, power, product, reshape, sin, where, zeros
from numpy.linalg import inv, norm, solve
from numpy.random import uniform, randn
from random import sample
from scipy.interpolate import interp1d
from keras.backend import constant
from ..controllers import PDController

def discrete_random_controller(m, sigma, t_eval):
    ts = t_eval[:-1]
    us = sigma * randn(len(ts), m)
    u_dict = {t: u for t, u in zip(ts, us)}

    def u_discrete_random(x, t):
        return u_dict[t]

    return u_discrete_random

def principal_scaling_connect_models(a, b):
    n, m = a.input_shape[-1], a.output_shape[-1]

    x = Input((n,))
    u_c = Input((m,))
    u_l = Input((m,))
    dVdx = Input((n,))
    g = Input((n, m))
    principal_scaling = Input((1,))

    a = a(x)
    b = b(x)

    u = Add()([u_c, u_l])
    known = Dot([1, 1])([dVdx, Dot([2, 1])([g, u_l])])
    unknown = Dot([1, 1])([principal_scaling, Add()([Dot([1, 1])([a, u]), b])])

    V_dot_r = Add()([known, unknown])

    return Model([dVdx, g, principal_scaling, x, u_c, u_l], V_dot_r)

def principal_scaling_cvx_augmenting_controller(u, V, LgV, dV, principal_scaling, a, b, alpha):
    a = evaluator(a)
    b = evaluator(b)

    C = 1e3

    def u_aug(x, t):
        u_c = u(x, t)
        u_l = Variable(u_c.shape)
        delta = Variable()
        obj = Minimize(1 / 2 * cvx_norm(u_c + u_l) ** 2 + C * delta ** 2)
        ps = principal_scaling(x, t)
        cons = [
            LgV(x, t) * u_l + ps * (a(x) * (u_c + u_l) + b(x)) + dV(x, u_c, t) + alpha * V(x, t) - delta <= 0,
            delta >= 0
        ]
        prob = Problem(obj, cons)
        prob.solve()
        return u_l.value

    return u_aug

def principal_scaling_augmenting_controller(u, V, LfV, LgV, dV, principal_scaling, a, b, C, alpha):
    a = evaluator(a)
    b = evaluator(b)

    def u_aug(x, t):
        u_c = u(x, t)

        lambda_r = (LfV(x, t) + principal_scaling(x, t) * b(x) + alpha * V(x, t)) /  (norm(LgV(x, t) + principal_scaling(x, t) * a(x)) ** 2 + 1 / C)
        lambda_r = max(0, lambda_r)
        lambda_plus = 0

        u_l = -u_c - lambda_r * (LgV(x, t) + principal_scaling(x, t) * a(x))
        delta = 1 / C * (lambda_r + lambda_plus)

        return u_l

    return u_aug

def weighted_controller(weight, u):

    def u_weighted(x, t):
        return weight * u(x, t)

    return u_weighted


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

def linear_model(d_in, output_shape, val):
    """Create a linear model (a one-layer neural network).

    Uses no activation. Initializes kernel to zero.

    Outputs keras model (R^d_in -> R^output_shape).

    Inputs:
    Input dimension, d_in: int
    Output shape, output_shape: int tuple
    Value to initialize layer bias with, val: float
    """
    model = Sequential()
    model.add(Dense(product(output_shape), input_shape=(d_in,), kernel_initializer = Constant(0), bias_initializer = Constant(val)))
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

def sparse_connect_models(a, b, n, m):
    """Connect two regression models a, b to form h(x, u) = dVdx * (g(x) * u_l + a(x) * (u_c + u_l) + b(x))
       after concatenating the output of a and b with the same sized zero arrays.

    x is in R^n and u_c, u_l are in R^m. Outputs keras model
    (R^n * R^(n * m) * R^n * R^m * R^m -> R).

    Inputs:
    Regression model, a: keras Sequential model (R^n -> R^(n/2 * m))
    Regression model, b: keras Sequential model (R^n -> R^n/2)
    """
    x, u_c, u_l = Input((n,)), Input((m,)), Input((m,))

    z_n, z_m = a.output_shape[1:3]
    z_a = Input((z_n,z_m))
    z_b = Input((z_n,))

    dVdx, g = Input((n,)), Input((n, m))
    a, b = a(x), b(x)

    a = Concatenate(axis = 1)([z_a, a])
    b = Concatenate(axis = 1)([z_b, b])

    gu_l = Dot([2, 1])([g, u_l])
    u = Add()([u_c, u_l])

    au = Dot([2, 1])([a, u])
    a_sum = Add()([gu_l, au, b])
    V_dot_r = Dot([1, 1])([dVdx, a_sum])
    model = Model([dVdx, g, x, u_c, u_l, z_a, z_b], V_dot_r)
    return model



def differentiator(L, h):
    """Create L-step causal differentiator filter.

    Outputs function mapping numpy array (N,) to numpy array (N - L + 1,).

    Inputs:
    Size of filter, L: int
    Sample time, h: float
    """

    # ks = reshape(arange(L), (L, 1))
    # A = power(ks.T, ks)
    # b = zeros(L)
    # b[1] = -1 / h
    # w = dot(inv(A), b)
    #
    # def diff(xs):
    #     return convolve(w, xs, 'valid')
    #
    # return diff

    half_L = (L - 1) // 2
    idxs = arange(-half_L, half_L + 1)
    pows = arange(0, L)
    A = array([idxs]) ** array([pows]).T
    b = zeros((L,))
    b[1] = 1 / h

    diff_kernel = solve(A, b)

    def diff(xs):
        return correlate(xs, diff_kernel, 'valid')

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
        Keras model history object, model_history: keras History object
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
    Conditions specified when creating the random perturbations, randPbConds: float * int * float * float
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

        model_history = model.fit([dVdxs, gs, xs, u_cs, u_ls], dV_r_hats, epochs=n_epochs, callbacks=callbacks_list, batch_size=len(xs)//10, validation_split=0.5)

        # save recently trained model and its components
        model.save('model.h5')
        a_model.save('a.h5')
        b_model.save('b.h5')

        C = 1e3
        u_l = augmenting_controller(dVdx, system.act, u_c, a_model, b_model, C)

        return a_model, b_model, model_history, u_l, u_ls_prev, epsilons_prev, xs, ts, sols

    return runEpisodes



def genRunEpPD(true_sys, true_sys_controller, K_pd, numstates, numdiff, numind, N, dt, num_timesteps, n_epochs, randPbConds, output_model_param, z_a_param):

    """
    Creates a function that learns augmentations to system dynamics based on data and/or a learning model from previous episodes

    Outputs function mapping
        System (with estimated parameters) object, est_sys: System
        Controller (with estimated parameters) object, est_sys_controller: Controller
        Regression model, a_model: keras Sequential model (R^n -> R^(n * m))
        Regression model, b_model: keras Sequential model (R^n -> R^n)
        Learned nominal controllers from previous episodes, u_cs_prev: (numpy array (n, ) * float -> numpy array (m,)) list
        Perturbations from previous episodes, u_ls_prev: ((numpy array (n,) * float -> numpy array (m,)) list) list
        State space data from previous episodes, existing_xdata: numpy array  (N * (i - 1), n)
        Time data from previous episodes, existing_tdata: numpy array (N * (i - 1),)
        Times and corresponding solutions from previous episodes, existing_solsdata: numpy array (N * (i - 1),) * numpy array (N * (i - 1), n)
        Initial conditions of trajectory for PD controller to follow, des_traj_ic: numpy array (n,)
        Initial state space conditions of trajectories generated by PD controller, x_0s: numpy array (N, n)
        Times corresponding to initial state space conditions of trajectories generated by PD controller, t_evals: numpy array (N,)
    to
        Updated regression model, a_model: keras Sequential model (R^n -> R^(n * m))
        Updated regression model, b_model: keras Sequential model (R^n -> R^n)
        Keras model history object, model_history: keras History object
        Updated learned nominal controllers from previous episodes, u_cs: (numpy array (n, ) * float -> numpy array (m,)) list
        Updated perturbations from previous episodes, u_ls: ((numpy array (n,) * float -> numpy array (m,)) list) list
        Updated state space data from previous episodes, xs: numpy array  (N * i, n)
        Updated time data from previous episodes, ts:  numpy array (N * i,)
        Updated times and corresponding solutions from previous episodes, sols: numpy array (N * i,) * numpy array (N * i, n)


    Inputs:
    System (with true values of parameters) object, true_Sys: system
    Controller (with true values of parameters) object true_sys_controller: Controller
    Gains of PD Controller, K_pd: numpy array (n, m)
    Number of states, numstates: int
    Length of differentiation filter, numdiff: int
    Number of points per trajectory to be trained on, numind: int
    Number of trajectories to generate per episode, N: int
    Sample time, dt: float
    Number of timesteps to simulate training data, num_timesteps: int
    Maximum number of epochs to run model, n_epochs: int
    Conditions specified when creating the random perturbations, randPbConds: float * int * float * float
    Size of combined model, output_model_param: int * int
    Size of a_model, z_a_param: int * int
    """

    A, num_sinusoids, lowerfb, upperfb = randPbConds
    timetosim = num_timesteps * dt
    output_model_n, output_model_m = output_model_param
    z_n, za_m = z_a_param
    dV_true = true_sys_controller.dV

    def runEpisodesPD(est_sys, est_sys_controller, a_model, b_model, u_cs_prev, u_ls_prev, existing_xdata, existing_tdata, existing_solsdata, des_traj_ic, x_0s, t_evals):
        u_qp, V, dVdx, dV = est_sys_controller.u, est_sys_controller.V, est_sys_controller.dVdx, est_sys_controller.dV

        # Simulate system with uncertain mass to get desired trajectory
        des_traj_time = linspace(0, timetosim, 1e3)
        t_ds, x_ds = est_sys.simulate(u_qp, des_traj_ic, des_traj_time)

        # interpolate resulting trajectory to get function
        r = interp1d(t_ds, x_ds[:, 0:numstates//2], axis=0)
        r_dot = interp1d(t_ds, x_ds[:, numstates//2:numstates], axis=0)

        # create PD controller
        pd_controller = PDController(K_pd, r, r_dot)
        u_pd = pd_controller.u

        # compile model to output final augmentations
        model = sparse_connect_models(a_model, b_model, output_model_n, output_model_m)
        model.compile('adam', 'mean_squared_error')

        diff = differentiator(numdiff, dt) # Differentiator filter

        # add perturbation to PD controller
        epsilons = [random_controller(A, num_sinusoids, lowerfb, upperfb, u_pd) for _ in range(N)]
        u_finals = [sum_controller([u_pd, epsilon]) for epsilon in epsilons]

        # Generate new training data using PD controller
        dataset = [true_sys.simulate(u_final, x_0, t_eval) for u_final, x_0, t_eval in zip(u_finals, x_0s, t_evals)]

        # get all points on trajectory/ corresponding times, epsilons, solutions
        rand_indexes = [sample(range(numdiff - 1, num_timesteps), numind) for _ in range(N)]
        newxs = [xs[rand_index] for idx, (_, xs) in enumerate(dataset) for rand_index in rand_indexes[idx] ]
        newts = [ts[rand_index] for idx, (ts, _) in enumerate(dataset) for rand_index in rand_indexes[idx] ]
        neweps = [epsilons[idx] for idx, _ in enumerate(dataset) for _ in rand_indexes[idx] ]
        sols = [(ts[rand_index - numdiff + 1:rand_index + 1], xs[rand_index - numdiff + 1:rand_index + 1]) for idx, (ts, xs) in enumerate(dataset) for rand_index in rand_indexes[idx] ]

        new_u_ls = array([epsilon(x, t) for epsilon, x, t in zip(neweps, newxs, newts)])
        new_u_cs = array([u_pd(x, t) for _, x, t in zip(neweps, newxs, newts)])

        # append recently generated training data to existing training data if it exists
        xs = concatenate((existing_xdata, array(newxs)))
        ts = concatenate((existing_tdata, array(newts)))
        existing_solsdata.extend(sols)

        sols = existing_solsdata
        u_cs = concatenate((u_cs_prev, new_u_cs))
        u_ls = concatenate((u_ls_prev, new_u_ls))

        # get data necessary to train model
        dV_hats = concatenate([diff(array([V(x, t) for x, t in zip(xs, ts)])) for ts, xs in sols])
        dV_ds = array([dV(x, u_c, t) for x, u_c, t in zip(xs, u_cs, ts)])
        # dV_r_hats = dV_hats - dV_ds

        # the method used below to calculate dV_r_hats is a temporary fix
        dV_trues = array([dV_true(x, u_c, t) for x, u_c, t in zip(xs, u_cs, ts)])
        dV_r_hats = dV_trues - dV_ds

        dVdxs = array([dVdx(x, t) for x, t in zip(xs, ts)])
        gs = array([est_sys.act(x) for x in xs])

        z_a = zeros((len(xs), z_n, za_m)) # array to concatenate with output of a_model
        z_b = zeros((len(xs), z_n)) # array to concatenate with output of b_model
        model_history = model.fit([dVdxs, gs, xs, u_cs, u_ls, z_a, z_b], dV_r_hats, epochs=n_epochs, batch_size=len(xs), validation_split=0.5)

        # save recently trained model and its components
        model.save('model.h5')
        a_model.save('a.h5')
        b_model.save('b.h5')

        return a_model, b_model, model_history, u_cs, u_ls, xs, ts, sols
    return runEpisodesPD

def interpolate(ts, xs, x_dots):

    def interp(t):
        before = where(ts <= t)[0]
        after = where(ts > t)[0]

        if len(after) == 0:
            idx_0 = before[-2]
            idx_1 = before[-1]
        else:
            idx_0 = before[-1]
            idx_1 = after[0]

        t_0, x_0, x_dot_0 = ts[idx_0], xs[idx_0], x_dots[idx_0]
        t_1, x_1, x_dot_1 = ts[idx_1], xs[idx_1], x_dots[idx_1]

        A = array([
            [t_0 ** 3, t_0 ** 2, t_0, 1],
            [t_1 ** 3, t_1 ** 2, t_1, 1],
            [3 * (t_0 ** 2), 2 * t_0, 1, 0],
            [3 * (t_1 ** 2), 2 * t_1, 1, 0]
        ])

        bs = array([x_0, x_1, x_dot_0, x_dot_1])

        alphas_0 = solve(A, bs)
        alphas_1 = array([3 * alphas_0[0], 2 * alphas_0[1], alphas_0[2]])
        alphas_2 = array([2 * alphas_1[0], alphas_1[1]])

        ts_0 = t ** arange(3, -1, -1)
        ts_1 = ts_0[1:]
        ts_2 = ts_1[1:]

        x_d = dot(ts_0, alphas_0)
        x_dot_d = dot(ts_1, alphas_1)
        x_ddot_d = dot(ts_2, alphas_2)

        return x_d, x_dot_d, x_ddot_d

    def r(t):
        return interp(t)[0]

    def r_dot(t):
        return interp(t)[1]

    def r_ddot(t):
        return interp(t)[2]

    return r, r_dot, r_ddot
