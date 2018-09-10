from matplotlib.pyplot import figure, grid, legend, plot, show
from numpy import array

from ..controllers import SegwayController
from ..systems import Segway

m_b, m_w, J_w, c_2, B_2, R, K, r, g, h, m_p = 20.42, 2.539, 0.063, -0.029, 0.578, 0.8796, 1.229, 0.195, 9.81, 0.56, 3.8

segway_est = Segway(m_b, m_w, J_w, c_2, B_2, R, K, r, g, h, m_p)
segway_true = Segway(m_b, m_w, J_w, c_2, B_2, R, K, r, g, h, m_p)

k_p, k_d = 100, 100
K = array([k_p, k_d])
r = lambda t: 0
r_dot, r_ddot = r, r

segway_controller = SegwayController(segway_est, K, r, r_dot, r_ddot)

u = segway_controller.u
x_0 = array([0, 0.1, 0, 0])
t_span = [0, 10]
dt = 1e-2
t_eval = [step * dt for step in range((t_span[-1] - t_span[0]) * int(1 / dt))]

ts, xs = segway_true.simulate(u, x_0, t_eval)

figure()
plot(ts, xs, linewidth=2)
grid()
legend(['$x$', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$'], fontsize=16)
show()
