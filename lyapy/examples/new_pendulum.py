from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import array, cumsum
from numpy.random import rand
from scipy.io import loadmat

from ..controllers import PDController, PendulumController
from ..systems import Pendulum
from ..learning import interpolate

m_hat, g, l_hat = 0.25, 9.81, 0.5
param_hats = array([m_hat, l_hat])
delta = 0.1
m, l = (2 * delta * rand(len(param_hats)) + 1 - delta) * param_hats

pendulum_true = Pendulum(m, g, l)
pendulum_est = Pendulum(m_hat, g, l_hat)

K_qp = array([1, 1])
K_pd = array([[20, 5]])

res = loadmat('./lyapy/trajectories/pendulum.mat')
x_ds, t_ds, u_ds = res['X_d'], res['T_d'][:, 0], res['U_d']

r, r_dot, r_ddot = interpolate(t_ds, x_ds[:, 0:1], x_ds[:, 1:])
r_qp = lambda t: r(t)[0]
r_dot_qp = lambda t: r_dot(t)[0]
r_ddot_qp = lambda t: r_ddot(t)[0]

qp_controller = PendulumController(pendulum_est, K_qp, r_qp, r_dot_qp, r_ddot_qp)
pd_controller = PDController(K_pd, r, r_dot)

x_0 = array([1, 0])
t_span = [0, 5]
dt = 1e-2
t_eval = [step * dt for step in range((t_span[-1] - t_span[0]) * int(1 / dt))]

t_qps, x_qps = pendulum_true.simulate(qp_controller.u, x_0, t_eval)
t_pds, x_pds = pendulum_true.simulate(pd_controller.u, x_0, t_eval)

u_qps = array([qp_controller.u(x, t) for x, t in zip(x_qps, t_qps)])
u_pds = array([pd_controller.u(x, t) for x, t in zip(x_pds, t_pds)])

int_u_qps = cumsum(abs(u_qps)) * dt
int_u_pds = cumsum(abs(u_pds)) * dt

figure()
suptitle('QP Controller', fontsize=16)

subplot(2, 1, 1)
plot(t_ds, x_ds[:, 0], '--', linewidth=2)
plot(t_qps, x_qps[:, 0], linewidth=2)
grid()
legend(['$\\theta_d$', '$\\theta$'], fontsize=16)

subplot(2, 1, 2)
plot(t_ds, x_ds[:, 1], '--', linewidth=2)
plot(t_qps, x_qps[:, 1], linewidth=2)
grid()
legend(['$\\dot{\\theta}_d$', '$\\dot{\\theta}$'], fontsize=16)

figure()
suptitle('PD Controller', fontsize=16)

subplot(2, 1, 1)
plot(t_ds, x_ds[:, 0], '--', linewidth=2)
plot(t_pds, x_pds[:, 0], linewidth=2)
grid()
legend(['$\\theta_d$', '$\\theta$'], fontsize=16)

subplot(2, 1, 2)
plot(t_ds, x_ds[:, 1], '--', linewidth=2)
plot(t_pds, x_pds[:, 1], linewidth=2)
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

show()


show()
