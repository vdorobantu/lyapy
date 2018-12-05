from keras.layers import Add, Dot, Input
from keras.models import Model
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import append, argsort, array, concatenate, cumsum, dot, exp, linspace, max, min, reshape, sign, split, zeros
from numpy.linalg import norm, lstsq
from numpy.random import rand
from scipy.io import loadmat

from ..controllers import PDController, SegwayController
from ..systems import Segway
from ..learning import constant_controller, differentiator, evaluator, interpolate, discrete_random_controller, sum_controller, two_layer_nn, principal_scaling_connect_models, principal_scaling_augmenting_controller, weighted_controller, fixed_connect_models, fixed_augmenting_controller

n, m = 4, 1

m_b_hat, m_w_hat, J_w_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, g, h_hat, m_p_hat = 20.42, 2.539, 0.063, -0.029, 0.578, 0.8796, 1.229, 0.195, 9.81, 0.56, 3.8
param_hats = array([m_b_hat, m_w_hat, J_w_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, h_hat, m_p_hat])
delta = 0.3
factors = 2 * delta * rand(len(param_hats)) + 1 - delta
m_b, m_w, J_w, c_2, B_2, R, K, r, h, m_p = factors * param_hats

segway_true = Segway(m_b, m_w, J_w, c_2, B_2, R, K, r, g, h, m_p)
segway_est = Segway(m_b_hat, m_w_hat, J_w_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, g, h_hat, m_p_hat)

K_qp = array([1, 1])
K_pd = array([[0, -100, 0, -1]])

res = loadmat('./lyapy/trajectories/segway.mat')
x_ds, t_ds, u_ds = res['X_d'], res['T_d'][:, 0], res['U_d']

r, r_dot, r_ddot = interpolate(t_ds, x_ds[:, 0:2], x_ds[:, 2:])
r_qp = lambda t: r(t)[1]
r_dot_qp = lambda t: r_dot(t)[1]
r_ddot_qp = lambda t: r_ddot(t)[1]

qp_controller = SegwayController(segway_est, K_qp, r_qp, r_dot_qp, r_ddot_qp)
true_controller = SegwayController(segway_true, K_qp, r_qp, r_dot_qp, r_ddot_qp)
pd_controller = PDController(K_pd, r, r_dot)

x_0 = array([2, 0, 0, 0])
t_span = [0, 5]
dt = 1e-3
t_eval = [step * dt for step in range((t_span[-1] - t_span[0]) * int(1 / dt))]

t_qps, x_qps = segway_true.simulate(qp_controller.u, x_0, t_eval)
width = 0.1
reps = 10
u_pd = sum_controller([pd_controller.u, discrete_random_controller(pd_controller.u, m, width, t_eval, reps)])
t_pds, x_pds = segway_true.simulate(u_pd, x_0, t_eval)

u_qps = array([qp_controller.u(x, t) for x, t in zip(x_qps, t_qps)])
u_pds = array([u_pd(x, t) for x, t in zip(x_pds, t_pds)])

int_u_qps = cumsum(abs(u_qps)) * dt
int_u_pds = cumsum(abs(u_pds)) * dt

figure()
suptitle('QP Controller', fontsize=16)

subplot(2, 2, 1)
plot(t_ds, x_ds[:, 0], '--', linewidth=2)
plot(t_qps, x_qps[:, 0], linewidth=2)
grid()
legend(['$x_d$', '$x$'], fontsize=16)

subplot(2, 2, 2)
plot(t_ds, x_ds[:, 1], '--', linewidth=2)
plot(t_qps, x_qps[:, 1], linewidth=2)
grid()
legend(['$\\theta_d$', '$\\theta$'], fontsize=16)

subplot(2, 2, 3)
plot(t_ds, x_ds[:, 2], '--', linewidth=2)
plot(t_qps, x_qps[:, 2], linewidth=2)
grid()
legend(['$\\dot{x}_d$', '$\\dot{x}$'], fontsize=16)

subplot(2, 2, 4)
plot(t_ds, x_ds[:, 3], '--', linewidth=2)
plot(t_qps, x_qps[:, 3], linewidth=2)
grid()
legend(['$\\dot{\\theta}_d$', '$\\dot{\\theta}$'], fontsize=16)

figure()
suptitle('PD Controller', fontsize=16)

subplot(2, 2, 1)
plot(t_ds, x_ds[:, 0], '--', linewidth=2)
plot(t_pds, x_pds[:, 0], linewidth=2)
grid()
legend(['$x_d$', '$x$'], fontsize=16)

subplot(2, 2, 2)
plot(t_ds, x_ds[:, 1], '--', linewidth=2)
plot(t_pds, x_pds[:, 1], linewidth=2)
grid()
legend(['$\\theta_d$', '$\\theta$'], fontsize=16)

subplot(2, 2, 3)
plot(t_ds, x_ds[:, 2], '--', linewidth=2)
plot(t_pds, x_pds[:, 2], linewidth=2)
grid()
legend(['$\\dot{x}_d$', '$\\dot{x}$'], fontsize=16)

subplot(2, 2, 4)
plot(t_ds, x_ds[:, 3], '--', linewidth=2)
plot(t_pds, x_pds[:, 3], linewidth=2)
grid()
legend(['$\\dot{\\theta}_d$', '$\\dot{\\theta}$'], fontsize=16)

figure()

subplot(2, 1, 1)
title('Control', fontsize=16)
plot(t_pds, u_pds, '--', linewidth=2)
plot(t_qps, u_qps, linewidth=2)
grid()
legend(['$u_{PD}$', '$u_{QP}$'], fontsize=16)


subplot(2, 1, 2)
title('Control integral', fontsize=16)
plot(t_pds, int_u_pds, '--', linewidth=2)
plot(t_qps, int_u_qps, linewidth=2)
grid()
legend(['$u_{PD}$', '$u_{QP}$'], fontsize=16)

d_hidden = 200

L = 3
diff = differentiator(L, dt)

num_episodes = 10
# weights = linspace(0, 1, num_episodes + 1)[:-1]
weights = 1 - linspace(0, 1, num_episodes + 1)[-1:0:-1] ** 2
# weights = [0]
num_trajectories = 1
num_pre_train_epochs = 5000
num_epochs = 5000
subsample_rate = reps

# principal_scaling = lambda x, t: qp_controller.dVdx(x, t)[-1]
alpha = 1 / qp_controller.lambda_1

dVdx_episodes = zeros((0, n))
g_episodes = zeros((0, n, m))
# principal_scaling_episodes = zeros((0,))
x_episodes = zeros((0, n))
u_c_episodes = zeros((0, m))
u_l_episodes = zeros((0, m))
V_r_dot_episodes = zeros((0,))
t_episodes = zeros((0,))

x_compares = []
u_compares = []
t_compares = []
u_int_compares = []

# window = 5
#
# x_lst_sq_episodes = zeros((0, n))
# a_lst_sq_episodes = zeros((0, m))
# b_lst_sq_episodes = zeros((0, 1))
# principal_scaling_lst_sq_episodes = zeros((0,))
# t_lst_sq_episodes = zeros((0,))

u_aug = constant_controller(zeros((m,)))

for episode, weight in enumerate(weights):
    print('EPISODE', episode + 1)

    a = two_layer_nn(n + 1, d_hidden, (m,))
    b = two_layer_nn(n + 1, d_hidden, (1,))
    # model = principal_scaling_connect_models(a, b)
    model = fixed_connect_models(a, b, n)

    a.compile('adam', 'mean_absolute_error')
    b.compile('adam', 'mean_absolute_error')
    model.compile('adam', 'mean_squared_error')

    u_c = sum_controller([pd_controller.u, weighted_controller(weight, u_aug)])
    u_ls = [discrete_random_controller(pd_controller.u, m, width, t_eval, reps) for _ in range(num_trajectories)]
    us = [sum_controller([u_c, u_l]) for u_l in u_ls]

    sols = [segway_true.simulate(u, x_0, t_eval) for u in us]

    Vs = [array([qp_controller.V(x, t) for x, t in zip(xs, ts)]) for ts, xs in sols]
    # TODO: FIX THIS
    # V_dots = [diff(Vs)[::subsample_rate] for Vs in Vs]

    half_L = (L - 1) // 2
    xs = [xs[half_L:-half_L:subsample_rate] for _, xs in sols]
    ts = [ts[half_L:-half_L:subsample_rate] for ts, _ in sols]
    u_cs = [array([u_c(x, t) for x, t in zip(xs, ts)]) for xs, ts in zip(xs, ts)]
    u_ls = [array([u_l(x, t) for x, t in zip(xs, ts)]) for xs, ts, u_l in zip(xs, ts, u_ls)]
    # TODO: FIX THIS TOO
    V_dots = [array([true_controller.dV(x, u_c + u_l, t) for x, u_c, u_l, t in zip(xs, u_cs, u_ls, ts)]) for xs, u_cs, u_ls, ts in zip(xs, u_cs, u_ls, ts)]

    V_d_dots = [array([qp_controller.dV(x, u_c, t) for x, u_c, t in zip(xs, u_cs, ts)]) for xs, u_cs, ts in zip(xs, u_cs, ts)]
    V_r_dots = [V_dots - V_d_dots for V_dots, V_d_dots in zip(V_dots, V_d_dots)]


    dVdxs = [array([qp_controller.dVdx(x, t) for x, t in zip(xs, ts)]) for xs, ts in zip(xs, ts)]
    # principal_scalings = [array([principal_scaling(x, t) for x, t in zip(xs, ts)]) for xs, ts in zip(xs, ts)]

    # A_lst_sqs = [array([principal_scaling * append(u_c + u_l, 1) for principal_scaling, u_c, u_l in zip(principal_scalings, u_cs, u_ls)]) for principal_scalings, u_cs, u_ls in zip(principal_scalings, u_cs, u_ls)]
    # b_lst_sqs = [array([V_r_dot - dot(qp_controller.LgV(x, t), u_l) for V_r_dot, x, u_l, t in zip(V_r_dots, xs, u_ls, ts)]) for V_r_dots, xs, u_ls, ts in zip(V_r_dots, xs, u_ls, ts)]
    #
    # w_lst_sqs = concatenate([array([lstsq(A_lst_sqs[idx:idx+window], b_lst_sqs[idx:idx+window], rcond=None)[0] for idx in range(len(A_lst_sqs) - window + 1)]) for A_lst_sqs, b_lst_sqs in zip(A_lst_sqs, b_lst_sqs)])
    # a_lst_sqs = w_lst_sqs[:, :-1]
    # b_lst_sqs = w_lst_sqs[:, -1:]
    #
    # half_window = (window - 1) // 2
    # x_lst_sqs = concatenate([xs[half_window:-half_window] for xs in xs])
    # principal_scaling_lst_sqs = concatenate([principal_scalings[half_window:-half_window] for principal_scalings in principal_scalings])
    # t_lst_sqs = concatenate([ts[half_window:-half_window] for ts in ts])
    #
    # x_lst_sq_episodes = concatenate([x_lst_sq_episodes, x_lst_sqs])
    # a_lst_sq_episodes = concatenate([a_lst_sq_episodes, a_lst_sqs])
    # b_lst_sq_episodes = concatenate([b_lst_sq_episodes, b_lst_sqs])
    # principal_scaling_lst_sq_episodes = concatenate([principal_scaling_lst_sq_episodes, principal_scaling_lst_sqs])
    # t_lst_sq_episodes = concatenate([t_lst_sq_episodes, t_lst_sqs])

    # N = len(x_lst_sq_episodes)

    # a_pre_trues = array([(true_controller.LgV(x, t) - qp_controller.LgV(x, t)) / principal_scaling for x, t, principal_scaling in zip(x_lst_sq_episodes, t_lst_sq_episodes, principal_scaling_lst_sq_episodes)])
    # b_pre_trues = array([(true_controller.LfV(x, t) - qp_controller.LfV(x, t)) / principal_scaling for x, t, principal_scaling in zip(x_lst_sq_episodes, t_lst_sq_episodes, principal_scaling_lst_sq_episodes)])

    # print('Fitting a...')
    # a.fit(x_lst_sq_episodes, a_lst_sq_episodes, epochs=num_pre_train_epochs, batch_size=N, validation_split=0.2)
    # a.fit(x_lst_sq_episodes, a_pre_trues, epochs=num_pre_train_epochs, batch_size=N, validation_split=0.2)
    # print('Fitting b...')
    # b.fit(x_lst_sq_episodes, b_lst_sq_episodes, epochs=num_pre_train_epochs, batch_size=N, validation_split=0.2)
    # b.fit(x_lst_sq_episodes, b_pre_trues, epochs=num_pre_train_epochs, batch_size=N, validation_split=0.2)

    # a_w1, a_b1 = a.layers[0].get_weights()
    # a_w2, a_b2 = a.layers[2].get_weights()
    # b_w1, b_b1 = b.layers[0].get_weights()
    # b_w2, b_b2 = b.layers[2].get_weights()
    #
    # print('a layer 1')
    # print(norm(a_w1, 2), norm(a_b1))
    # print('a layer 2')
    # print(norm(a_w2, 2), norm(a_b2))
    # print('b layer 1')
    # print(norm(b_w1, 2), norm(b_b1))
    # print('b layer 2')
    # print(norm(b_w2, 2), norm(b_b2))
    #
    # a_pre_ests = a.predict(x_lst_sq_episodes)
    # a_pre_ests = array([principal_scaling * a_est for principal_scaling, a_est in zip(principal_scaling_lst_sq_episodes, a_pre_ests)])
    # b_pre_ests = b.predict(x_lst_sq_episodes)[:,0]
    # b_pre_ests = array([principal_scaling * b_est for principal_scaling, b_est in zip(principal_scaling_lst_sq_episodes, b_pre_ests)])
    #
    # a_pre_trues = array([true_controller.LgV(x, t) - qp_controller.LgV(x, t) for x, t in zip(x_lst_sq_episodes, t_lst_sq_episodes)])
    # b_pre_trues = array([true_controller.LfV(x, t) - qp_controller.LfV(x, t) for x, t in zip(x_lst_sq_episodes, t_lst_sq_episodes)])
    # a_mse = norm(a_pre_ests - a_pre_trues, 'fro') ** 2 / (2 * N)
    # b_mse = norm(b_pre_ests - b_pre_trues) ** 2 / (2 * N)
    # print('a_mse', a_mse, 'b_mse', b_mse)

    dVdxs = concatenate(dVdxs)
    # principal_scalings = concatenate(principal_scalings)
    xs = concatenate(xs)
    u_cs = concatenate(u_cs)
    u_ls = concatenate(u_ls)
    V_r_dots = concatenate(V_r_dots)

    ts = concatenate(ts)

    gs = array([segway_est.act(x) for x in xs])

    dVdx_episodes = concatenate([dVdx_episodes, dVdxs])
    g_episodes = concatenate([g_episodes, gs])
    # principal_scaling_episodes = concatenate([principal_scaling_episodes, principal_scalings])
    x_episodes = concatenate([x_episodes, xs])
    u_c_episodes = concatenate([u_c_episodes, u_cs])
    u_l_episodes = concatenate([u_l_episodes, u_ls])
    V_r_dot_episodes = concatenate([V_r_dot_episodes, V_r_dots])
    t_episodes = concatenate([t_episodes, ts])

    input_episodes = concatenate([x_episodes, dVdx_episodes[:, -1:]], 1)
    beta = 0.01
    sample_weight_episodes = exp(-beta * norm(dVdx_episodes[:, -1:], axis=1))

    N = len(x_episodes)

    a_trues = array([true_controller.LgV(x, t) - qp_controller.LgV(x, t) for x, t in zip(x_episodes, t_episodes)])
    b_trues = array([true_controller.LfV(x, t) - qp_controller.LfV(x, t) for x, t in zip(x_episodes, t_episodes)])

    print('Pretraining a...')
    a.fit(input_episodes, a_trues, epochs=num_pre_train_epochs, batch_size=N, sample_weight=sample_weight_episodes)

    print('Pretraining b...')
    b.fit(input_episodes, b_trues, epochs=num_pre_train_epochs, batch_size=N, sample_weight=sample_weight_episodes)

    print('Fitting V_r_dot...')
    # model.fit([dVdx_episodes, g_episodes, principal_scaling_episodes, x_episodes, u_c_episodes, u_l_episodes], V_r_dot_episodes, epochs=num_epochs, batch_size=N)
    model.fit([dVdx_episodes, g_episodes, input_episodes, u_c_episodes, u_l_episodes], V_r_dot_episodes, epochs=num_epochs, batch_size=N, sample_weight=sample_weight_episodes)

    # a_post_ests = a.predict(x_episodes)
    # a_post_ests = array([principal_scaling * a_est for principal_scaling, a_est in zip(principal_scaling_episodes, a_post_ests)])
    # b_post_ests = b.predict(x_episodes)[:,0]
    # b_post_ests = array([principal_scaling * b_est for principal_scaling, b_est in zip(principal_scaling_episodes, b_post_ests)])
    #
    # a_post_trues = array([true_controller.LgV(x, t) - qp_controller.LgV(x, t) for x, t in zip(x_episodes, t_episodes)])
    # b_post_trues = array([true_controller.LfV(x, t) - qp_controller.LfV(x, t) for x, t in zip(x_episodes, t_episodes)])
    # a_mse = norm(a_post_ests - a_post_trues, 'fro') ** 2 / (2 * N)
    # b_mse = norm(b_post_ests - b_post_trues) ** 2 / (2 * N)
    # print('a_mse', a_mse, 'b_mse', b_mse)
    #
    # a_w1, a_b1 = a.layers[0].get_weights()
    # a_w2, a_b2 = a.layers[2].get_weights()
    # b_w1, b_b1 = b.layers[0].get_weights()
    # b_w2, b_b2 = b.layers[2].get_weights()
    #
    # print('a layer 1')
    # print(norm(a_w1, 2), norm(a_b1))
    # print('a layer 2')
    # print(norm(a_w2, 2), norm(a_b2))
    # print('b layer 1')
    # print(norm(b_w1, 2), norm(b_b1))
    # print('b layer 2')
    # print(norm(b_w2, 2), norm(b_b2))

    C = 1e4
    inp = lambda x, t: concatenate([x, qp_controller.dVdx(x, t)[-1:]])
    u_aug = fixed_augmenting_controller(pd_controller.u, inp, qp_controller.V, qp_controller.LfV, qp_controller.LgV, a, b, C, alpha)
    # u_aug = principal_scaling_augmenting_controller(pd_controller.u, qp_controller.V, qp_controller.LfV, qp_controller.LgV, qp_controller.dV, principal_scaling, a, b, C, alpha)

    ts, xs = segway_true.simulate(u_c, x_0, t_eval)
    us = array([u_c(x, t) for x, t in zip(xs, ts)])
    u_ints = cumsum(array([norm(u) for u in us])) * dt

    x_compares.append(xs)
    u_compares.append(us)
    t_compares.append(ts)
    u_int_compares.append(u_ints)

# a_lst_sq_episodes = array([a_lst_sq * principal_scaling for a_lst_sq, principal_scaling in zip(a_lst_sq_episodes, principal_scaling_lst_sq_episodes)])
# b_lst_sq_episodes = array([b_lst_sq * principal_scaling for b_lst_sq, principal_scaling in zip(b_lst_sq_episodes, principal_scaling_lst_sq_episodes)])
#
# figure()
# suptitle('$a$ and $b$ least squares estimates', fontsize=16)
#
# subplot(2, 1, 1)
# title('a')
# plot(a_lst_sq_episodes)
# plot(a_pre_trues, '--')
# grid()
#
# subplot(2, 1, 2)
# title('b')
# plot(b_lst_sq_episodes)
# plot(b_pre_trues, '--')
# grid()
#
# figure()
# suptitle('$a$ and $b$ pretrained estimates', fontsize=16)
#
# subplot(2, 1, 1)
# title('a')
# plot(a_pre_ests)
# plot(a_pre_trues, '--')
# grid()
#
# subplot(2, 1, 2)
# title('b')
# plot(b_pre_ests)
# plot(b_pre_trues, '--')
# grid()
#
# figure()
# suptitle('$a$ and $b$ trained estimates', fontsize=16)
#
# subplot(2, 1, 1)
# title('a')
# plot(a_post_ests)
# plot(a_post_trues, '--')
# grid()
#
# subplot(2, 1, 2)
# title('b')
# plot(b_post_ests)
# plot(b_post_trues, '--')
# grid()

# TODO: Remove this weighted thing
u = sum_controller([pd_controller.u, u_aug])
# u = sum_controller([pd_controller.u, weighted_controller(0.1, u_aug)])

ts, xs = segway_true.simulate(u, x_0, t_eval)
us = array([u(x, t) for x, t in zip(xs, ts)])
u_ints = cumsum(array([norm(u) for u in us])) * dt

x_compares.append(xs)
u_compares.append(us)
t_compares.append(ts)
u_int_compares.append(u_ints)

a_ests = a.predict(input_episodes)
b_ests = b.predict(input_episodes)

figure()
suptitle('Debug', fontsize=16)

subplot(2, 1, 1)
plot(a_ests, linewidth=2)
plot(a_trues, '--', linewidth=2)
grid()
legend(['Estimated', 'True'], fontsize=16)
title('a', fontsize=16)

subplot(2, 1, 2)
plot(b_ests, linewidth=2)
plot(b_trues, '--', linewidth=2)
grid()
legend(['Estimated', 'True'], fontsize=16)
title('b', fontsize=16)

figure()
suptitle('Augmented Controller', fontsize=16)

subplot(2, 2, 1)
plot(t_ds, x_ds[:, 0], '--', linewidth=2)
plot(ts, xs[:, 0], linewidth=2)
grid()
legend(['$x_d$', '$x$'], fontsize=16)

subplot(2, 2, 2)
plot(t_ds, x_ds[:, 1], '--', linewidth=2)
plot(ts, xs[:, 1], linewidth=2)
grid()
legend(['$\\theta_d$', '$\\theta$'], fontsize=16)

subplot(2, 2, 3)
plot(t_ds, x_ds[:, 2], '--', linewidth=2)
plot(ts, xs[:, 2], linewidth=2)
grid()
legend(['$\\dot{x}_d$', '$\\dot{x}$'], fontsize=16)

subplot(2, 2, 4)
plot(t_ds, x_ds[:, 3], '--', linewidth=2)
plot(ts, xs[:, 3], linewidth=2)
grid()
legend(['$\\dot{\\theta}_d$', '$\\dot{\\theta}$'], fontsize=16)

def V_r_dot_true(x, u_c, u_l, t):
    return true_controller.dV(x, u_c + u_l, t) - qp_controller.dV(x, u_c, t)

V_r_dot_true_episodes = array([V_r_dot_true(x, u_c, u_l, t) for x, u_c, u_l, t in zip(x_episodes, u_c_episodes, u_l_episodes, t_episodes)])

figure()
title('Derivative Estimation', fontsize=16)
plot(V_r_dot_episodes, linewidth=2)
plot(V_r_dot_true_episodes, '--', linewidth=2)
grid()
legend(['Estimated', 'Truth'], fontsize=16)

us = array([u(x, t) for x, t in zip(x_episodes, t_episodes)])
u_trues = array([true_controller.u(x, t) for x, t in zip(x_episodes, t_episodes)])

figure()
title('Controller comparison', fontsize=16)
plot(us, linewidth=2)
plot(u_trues, linewidth=2)
grid()
legend(['Augmented', 'Perfect'], fontsize=16)

Vs = array([qp_controller.V(x, t) for x, t in zip(xs, ts)])

t_trues, x_trues = segway_true.simulate(true_controller.u, x_0, t_eval)
V_trues = array([true_controller.V(x, t) for x, t in zip(x_trues, t_trues)])

figure()
title('Lyapunov function comparison', fontsize=16)
plot(ts, Vs, linewidth=2)
plot(ts, V_trues, '--', linewidth=2)
grid()
legend(['Augmented', 'Perfect'], fontsize=16)

show()

x_animates = split(x_episodes, num_episodes)
u_c_animates = split(u_c_episodes, num_episodes)
u_l_animates = split(u_l_episodes, num_episodes)
ts = t_eval[half_L:-half_L-1:subsample_rate]
t_0, t_final = ts[0], ts[-1]

x_min, theta_min, x_dot_min, theta_dot_min = min(concatenate(x_compares), 0)
x_max, theta_max, x_dot_max, theta_dot_max = max(concatenate(x_compares), 0)
u_min = min(concatenate(u_compares, 0)[:, 0])
u_max = max(concatenate(u_compares, 0)[:, 0])
u_int_min = min(concatenate(u_int_compares, 0))
u_int_max = max(concatenate(u_int_compares, 0))
mins = [x_min, theta_min, u_min, x_dot_min, theta_dot_min, u_int_min]
maxs = [x_max, theta_max, u_max, x_dot_max, theta_dot_max, u_int_max]

print(factors)

f = figure()
axs = f.subplots(2, 3)

ax_ds = reshape(axs[0:2, 0:2], -1)
for ax_d, traj_d in zip(ax_ds, x_ds.T):
    ax_d.plot(t_ds, traj_d, '--')

axs = reshape(axs, -1)
titles = ['$x$', '$\\theta$', '$u$', '$\\dot{x}$', '$\\dot{\\theta}$', '$\\int u$']

for ax, title, minimum, maximum in zip(axs, titles, mins, maxs):
    ax.set_xlim(t_0, t_final)
    ax.set_ylim(min([0.9 * minimum, 1.1 * minimum]), max([0.9 * maximum, 1.1 * maximum]))
    ax.set_title(title, fontsize=16)
    ax.grid()
lines = [ax.plot([], [], linewidth=2)[0] for ax in axs]

def update(frame):
    xs, thetas, x_dots, theta_dots = x_compares[frame].T
    us = u_compares[frame][:, 0]
    u_ints = u_int_compares[frame]

    trajs = [xs, thetas, us, x_dots, theta_dots, u_ints]

    for line, traj in zip(lines, trajs):
        line.set_data(t_compares[frame], traj)

    return lines

_ = FuncAnimation(f, update, frames=range(num_episodes + 1), blit=True, interval=500, repeat=True, repeat_delay=2000)

show(f)
