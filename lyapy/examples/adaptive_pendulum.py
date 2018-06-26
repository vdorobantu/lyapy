from matplotlib.pyplot import figure, grid, legend, plot, show, subplot
from numpy import array, concatenate, linspace

from ..controllers import AdaptivePendulumController
from ..systems import AdaptivePendulum

m, g, l, gamma = 0.25, 9.81, 0.5, 1
k_p, k_d = 1, 2
K = array([k_p, k_d])
adaptive_pendulum = AdaptivePendulum(m, g, l, K, gamma)
adaptive_pendulum_controller = AdaptivePendulumController(adaptive_pendulum)
u, _, _ = adaptive_pendulum_controller.synthesize()

m_hat = m * 0.8
x_0 = array([1, 0, 1 / m_hat])
t_eval = linspace(0, 10, 1e4)
ts, xs = adaptive_pendulum.simulate(u, x_0, t_eval)
us = concatenate([u(x, t) for x, t in zip(xs, ts)])

figure()
subplot(1, 2, 1)
plot(ts, xs[:, 0:2], linewidth=3)
grid()
legend(['$\\psi$', '$\\dot{\\psi}$'], fontsize=16)

subplot(1, 2, 2)
plot(ts, us, linewidth=3)
grid()
legend(['$u$'], fontsize=16)

figure()
plot(ts, xs[:, -1], linewidth=3)
grid()
legend(['$\\hat{\\theta}$'], fontsize=16)

show()
