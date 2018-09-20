from keras.layers import Add, Dot, Input
from keras.models import Model
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import array, concatenate, dot, cumsum, linspace, sign, zeros
from numpy.linalg import norm
from numpy.random import rand
from scipy.io import loadmat

from ..controllers import PDController, SegwayController
from ..systems import Segway
from ..learning import constant_controller, differentiator, evaluator, interpolate, discrete_random_controller, sum_controller, two_layer_nn, principal_scaling_connect_models, principal_scaling_augmenting_controller, weighted_controller

n, m = 4, 1

m_b_hat, m_w_hat, J_w_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, g, h_hat, m_p_hat = 20.42, 2.539, 0.063, -0.029, 0.578, 0.8796, 1.229, 0.195, 9.81, 0.56, 3.8
param_hats = array([m_b_hat, m_w_hat, J_w_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, h_hat, m_p_hat])
delta = 0.2
m_b, m_w, J_w, c_2, B_2, R, K, r, h, m_p = (2 * delta * rand(len(param_hats)) + 1 - delta) * param_hats

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
pd_controller = PDController(K_pd, r, r_dot)

x_0 = array([2, 0, 0, 0])
t_span = [0, 5]
dt = 1e-3
t_eval = [step * dt for step in range((t_span[-1] - t_span[0]) * int(1 / dt))]

t_qps, x_qps = segway_true.simulate(qp_controller.u, x_0, t_eval)
sigma = 1
u_pd = sum_controller([pd_controller.u, discrete_random_controller(m, sigma, t_eval)])
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

num_episodes = 20
weights = linspace(0, 1, num_episodes + 1)[:-1]
num_trajectories = 1
num_epochs = 100
subsample_rate = 10

principal_scaling = lambda x, t: norm(qp_controller.dVdx(x, t)[2:])
alpha = 1 / qp_controller.lambda_1

dVdx_episodes = zeros((0, n))
g_episodes = zeros((0, n, m))
principal_scaling_episodes = zeros((0,))
x_episodes = zeros((0, n))
u_c_episodes = zeros((0, m))
u_l_episodes = zeros((0, m))
V_r_dot_episodes = zeros((0,))

u_aug = constant_controller(zeros((m,)))

for episode, weight in enumerate(weights):
    print('EPISODE', episode + 1)

    a = two_layer_nn(n, d_hidden, (m,), 0.5)
    b = two_layer_nn(n, d_hidden, (1,), 0.5)

    model = principal_scaling_connect_models(a, b)
    model.compile('adam', 'mean_squared_error')

    u_c = sum_controller([pd_controller.u, weighted_controller(weight, u_aug)])
    u_ls = [discrete_random_controller(m, sigma, t_eval) for _ in range(num_trajectories)]
    us = [sum_controller([u_c, u_l]) for u_l in u_ls]

    sols = [segway_true.simulate(u, x_0, t_eval) for u in us]

    Vs = [array([qp_controller.V(x, t) for x, t in zip(xs, ts)]) for ts, xs in sols]
    V_dots = concatenate([diff(V) for V in Vs])

    half_L = (L - 1) // 2
    xs = concatenate([xs[half_L:-half_L] for _, xs in sols])
    ts = concatenate([ts[half_L:-half_L] for ts, _ in sols])
    u_cs = array([u_c(x, t) for x, t in zip(xs, ts)])
    u_ls = concatenate([array([u_l(x, t) for x, t in zip(xs, ts)])[half_L:-half_L] for (ts, xs), u_l in zip(sols, u_ls)])

    V_d_dots = array([qp_controller.dV(x, u_c, t) for x, u_c, t in zip(xs, u_cs, ts)])
    V_r_dots = V_dots - V_d_dots

    dVdxs = array([qp_controller.dVdx(x, t) for x, t in zip(xs, ts)])
    gs = array([segway_est.act(x) for x in xs])
    principal_scalings = array([norm(dVdx[2:]) for dVdx in dVdxs])

    dVdx_episodes = concatenate([dVdx_episodes, dVdxs[::subsample_rate]])
    g_episodes = concatenate([g_episodes, gs[::subsample_rate]])
    principal_scaling_episodes = concatenate([principal_scaling_episodes, principal_scalings[::subsample_rate]])
    x_episodes = concatenate([x_episodes, xs[::subsample_rate]])
    u_c_episodes = concatenate([u_c_episodes, u_cs[::subsample_rate]])
    u_l_episodes = concatenate([u_l_episodes, u_ls[::subsample_rate]])
    V_r_dot_episodes = concatenate([V_r_dot_episodes, V_r_dots[::subsample_rate]])

    N = len(x_episodes)

    model.fit([dVdx_episodes, g_episodes, principal_scaling_episodes, x_episodes, u_c_episodes, u_l_episodes], V_r_dot_episodes, epochs=num_epochs)
    a_ests = a.predict(x_episodes)
    a_ests = array([principal_scaling * a_est for principal_scaling, a_est in zip(principal_scaling_episodes, a_ests)])
    b_ests = b.predict(x_episodes)[:,0]
    b_ests = array([principal_scaling * b_est for principal_scaling, b_est in zip(principal_scaling_episodes, b_ests)])

    a_trues = array([dot(dVdx, segway_true.act(x) - segway_est.act(x)) for dVdx, x, in zip(dVdx_episodes, x_episodes)])
    b_trues = array([dot(dVdx, segway_true.drift(x) - segway_est.drift(x)) for dVdx, x, in zip(dVdx_episodes, x_episodes)])
    a_mse = norm(a_ests - a_trues, 'fro') ** 2 / (2 * N)
    b_mse = norm(b_ests - b_trues) ** 2 / (2 * N)
    print('a_mse', a_mse, 'b_mse', b_mse)

    C = 1e3
    u_aug = principal_scaling_augmenting_controller(pd_controller.u, qp_controller.V, qp_controller.LfV, qp_controller.LgV, qp_controller.dV, principal_scaling, a, b, C, alpha)

figure()

subplot(2, 1, 1)
title('a')
plot(a_ests)
plot(a_trues, '--')
grid()

subplot(2, 1, 2)
title('b')
plot(b_ests)
plot(b_trues, '--')
grid()

u = sum_controller([pd_controller.u, u_aug])

ts, xs = segway_true.simulate(u, x_0, t_eval)

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

show()
