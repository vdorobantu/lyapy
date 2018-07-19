"""Augmentation, simulation, and plotting for inverted pendulum control system."""

from matplotlib.pyplot import figure, grid, legend, plot, scatter, show, subplot, suptitle, xlabel, ylabel
from numpy import array, concatenate, cos, exp, linspace, pi, sin, sqrt
from numpy.linalg import eigvals
from numpy.random import rand, randn

from ..controllers import PendulumController
from ..learning import augmenting_controller, connect_models, constant_controller, differentiator, sum_controller, two_layer_nn
from ..systems import Pendulum

m, g, l = 0.25, 9.81, 0.5
pendulum = Pendulum(m, g, l)
m_hat, K = m * 0.8, array([1, 2])
A, T = 1, 5 # Amplitude, period of desired trajectory
omega = 2 * pi / T

def r(t):
    return A * cos(omega * t)

def r_dot(t):
    return -omega * A * sin(omega * t)

def r_ddot(t):
    return -(omega ** 2) * r(t)

pendulum_controller = PendulumController(pendulum, m_hat, K, r, r_dot, r_ddot)
u, V, dVdx, dV = pendulum_controller.u, pendulum_controller.V, pendulum_controller.dVdx, pendulum_controller.dV

# True Lyapunov function derivative for comparison
pendulum_controller_true = PendulumController(pendulum, m, K, r, r_dot, r_ddot)
dV_true = pendulum_controller_true.dV

# Single simulation
x_0 = array([1.5, 0])
t_eval = linspace(0, 10, 1e3)
ts, xs = pendulum.simulate(u, x_0, t_eval)

x_ds = array([array([r(t), r_dot(t)]) for t in ts])
us = array([u(x, t) for x, t in zip(xs, ts)])
Vs = array([V(x, t) for x, t in zip(xs, ts)])
dV_trues = array([dV_true(x, u, t) for x, u, t in zip(xs, us, ts)])

figure()
subplot(2, 2, 1)
plot(ts, xs, linewidth=3)
plot(ts, x_ds, '--', linewidth=3)
grid()
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=14)

subplot(2, 2, 2)
plot(ts, us, linewidth=3)
grid()
legend(['$u$'], fontsize=14)

subplot(2, 2, 3)
plot(ts, Vs, linewidth=3)
grid()
legend(['$V$'], fontsize=14)

subplot(2, 2, 4)
plot(ts, dV_trues, linewidth=3)
grid()
legend(['$\\dot{V}$'], fontsize=14)
suptitle('Nominal control system', fontsize=16)

# Lypaunov function derivative estimator
k, dropout_prob = 50, 0.5
a = two_layer_nn(2, k, (2, 1), dropout_prob)
b = two_layer_nn(2, k, (2,), dropout_prob)
model = connect_models(a, b)
model.compile('adam', 'mean_squared_error')

N, sigma, dt = 1000, sqrt(0.01), 1e-2 # Number of simulations, perturbation variance, sample time
diff = differentiator(2, dt) # Differentiator filter

u_ls = [constant_controller(u_0) for u_0 in sigma * randn(N, 1)] # Constant perturbations
u_augs = [sum_controller([u, u_l]) for u_l in u_ls] # Controller + constant perturbations

t_0s = 10 * rand(N, 1)
t_evals = t_0s + array([0, dt])
x_ds = array([array([r(t_0), r_dot(t_0)]) for t_0 in t_0s[:, 0]])
x_0s = x_ds + sqrt(0.1) * randn(N, 2)
# Simulating each perturbed controller from corresponding initial condition
sols = [pendulum.simulate(u_aug, x_0, t_eval) for u_aug, x_0, t_eval in zip(u_augs, x_0s, t_evals)]

xs = array([xs[-1] for _, xs in sols])
ts = array([ts[-1] for ts, _ in sols])
u_cs = array([u(x, t) for x, t in zip(xs, ts)])
u_ls = array([u_l(x, t) for x, u_l, t in zip(xs, u_ls, ts)])
# Numerically differentiating V for each simulation
dV_hats = concatenate([diff(array([V(x, t) for x, t in zip(xs, ts)])) for ts, xs in sols])
dV_ds = array([dV(x, u_c, t) for x, u_c, u_l, t in zip(xs, u_cs, u_ls, ts)])
dV_r_hats = dV_hats - dV_ds

dVdxs = array([dVdx(x, t) for x, t in zip(xs, ts)])
gs = array([pendulum.act(x) for x in xs])

figure()
suptitle('State space samples', fontsize=16)
scatter(xs[:, 0], xs[:, 1])
grid()
xlabel('$\\theta$', fontsize=16)
ylabel('$\\dot{\\theta}$', fontsize=16)

model.fit([dVdxs, gs, xs, u_cs, u_ls], dV_r_hats, epochs=200, batch_size=N // 10, validation_split=0.5)

C = 1e3
u_l = augmenting_controller(dVdx, pendulum.act, u, a, b, C)
u_aug = sum_controller([u, u_l])

# Single simulation
x_0 = array([1.5, 0])
t_eval = linspace(0, 10, 1e3)
ts, xs = pendulum.simulate(u_aug, x_0, t_eval)

x_ds = array([array([r(t), r_dot(t)]) for t in ts])
u_augs = array([u_aug(x, t) for x, t in zip(xs, ts)])
Vs = array([V(x, t) for x, t in zip(xs, ts)])
dV_trues = array([dV_true(x, u_aug, t) for x, u_aug, t in zip(xs, u_augs, ts)])
dVs = array([dV(x, u_aug, t) for x, u_aug, t in zip(xs, u_augs, ts)])

# Exponential upper bound on V(x(t))
lambda_1 = max(eigvals(pendulum_controller.P))
envelope = V(xs[0], ts[0]) * exp(-1 / lambda_1 * ts)

figure()
subplot(2, 2, 1)
plot(ts, xs, linewidth=3)
plot(ts, x_ds, '--', linewidth=3)
grid()
legend(['$\\theta$', '$\\dot{\\theta}$', '$\\theta_d$', '$\\dot{\\theta}_d$'], fontsize=14)

subplot(2, 2, 2)
plot(ts, u_augs, linewidth=3)
grid()
legend(['$u$'], fontsize=14)

subplot(2, 2, 3)
plot(ts, Vs, linewidth=3)
plot(ts, envelope, '--', linewidth=3)
grid()
legend(['$V$', 'Exp bound'], fontsize=14)

subplot(2, 2, 4)
plot(ts, dV_trues, linewidth=3)
grid()
legend(['$\\dot{V}$'], fontsize=14)
suptitle('Augmented control system', fontsize=16)

show()
