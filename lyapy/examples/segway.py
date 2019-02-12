"""Planar segway example."""

from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import arange, array, concatenate, cos, identity, ones, sin, tanh
from numpy.random import uniform
from scipy.io import loadmat

from ..controllers import CombinedController, PDController, QPController
from ..learning import decay_widths, evaluator, KerasTrainer, sigmoid_weighting, SimulationHandler
from ..lyapunov_functions import QuadraticControlLyapunovFunction
from ..outputs import RoboticSystemOutput
from ..systems import AffineControlSystem

class SegwaySystem(AffineControlSystem):
    def __init__(self, m_b=44.798, m_w=2.485, J_w=0.055936595310797, a_2=-0.023227187592750, c_2=0.166845864363019, B_2=2.899458828344427, R= 0.086985141514373, K=0.141344665167821, r=0.195, g=9.81, f_d=0.076067344020759, f_v=0.002862586216301, V_nom=57):

        self.f_3 = lambda x_dot, theta, theta_dot: (1/2) * R ** (-1) * (4 * \
        B_2 * J_w + 4 * a_2 ** 2 * J_w * m_b + 4 * c_2 ** 2 * J_w * m_b + 2 * \
        B_2 * m_b * r ** 2 + a_2 ** 2 * m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** \
        2 * r ** 2 + 4 * B_2 * m_w * r ** 2 + 4 * a_2 ** 2 * m_b * m_w * r ** \
        2 + 4 * c_2 ** 2 * m_b * m_w * r ** 2 + (a_2 ** 2 + (-1) * c_2 ** 2) * \
        m_b ** 2 * r ** 2 * cos(2 * theta) + 2 * a_2 * c_2 * m_b ** 2 * r ** 2 \
        * sin(2 * theta)) ** (-1) * (800 * B_2 * K ** 2 * theta_dot * r + 800 \
        * a_2 ** 2 * K ** 2 * m_b * theta_dot * r + 800 * c_2 ** 2 * K ** 2 * \
        m_b * theta_dot * r + 800 * B_2 * f_v * theta_dot * r * R + 800 * a_2 \
        ** 2 * f_v * m_b * theta_dot * r * R + 800 * c_2 ** 2 * f_v * m_b * \
        theta_dot * r * R + (-800) * B_2 * K ** 2 * x_dot + (-800) * a_2 ** 2 \
        * K ** 2 * m_b * x_dot + (-800) * c_2 ** 2 * K ** 2 * m_b * x_dot + \
        (-800) * B_2 * f_v * R * x_dot + (-800) * a_2 ** 2 * f_v * m_b * R * \
        x_dot + (-800) * c_2 ** 2 * f_v * m_b * R * x_dot + 80 * c_2 * K ** 2 \
        * m_b * theta_dot * r ** 2 * cos(theta) + 80 * c_2 * f_v * m_b * \
        theta_dot * r ** 2 * R * cos(theta) + 4 * a_2 * B_2 * m_b * theta_dot \
        ** 2 * r ** 2 * R * cos(theta) + 4 * a_2 ** 3 * m_b ** 2 * theta_dot \
        ** 2 * r ** 2 * R * cos(theta) + 4 * a_2 * c_2 ** 2 * m_b ** 2 * \
        theta_dot ** 2 * r ** 2 * R * cos(theta) + (-80) * c_2 * K ** 2 * m_b \
        * r * x_dot * cos(theta) + (-80) * c_2 * f_v * m_b * r * R * x_dot * \
        cos(theta) + (-4) * a_2 * c_2 * g * m_b ** 2 * r ** 2 * R * cos(2 * \
        theta) + (-80) * a_2 * K ** 2 * m_b * theta_dot * r ** 2 * sin(theta) \
        + (-80) * a_2 * f_v * m_b * theta_dot * r ** 2 * R * sin(theta) + 4 * \
        B_2 * c_2 * m_b * theta_dot ** 2 * r ** 2 * R * sin(theta) + 4 * a_2 \
        ** 2 * c_2 * m_b ** 2 * theta_dot ** 2 * r ** 2 * R * sin(theta) + 4 * \
        c_2 ** 3 * m_b ** 2 * theta_dot ** 2 * r ** 2 * R * sin(theta) + 80 * \
        a_2 * K ** 2 * m_b * r * x_dot * sin(theta) + 80 * a_2 * f_v * m_b * r \
        * R * x_dot * sin(theta) + 2 * a_2 ** 2 * g * m_b ** 2 * r ** 2 * R * \
        sin(2 * theta) + (-2) * c_2 ** 2 * g * m_b ** 2 * r ** 2 * R * sin(2 * \
        theta) + 4 * f_d * r * R * (10 * (B_2 + (a_2 ** 2 + c_2 ** 2) * m_b) + \
        c_2 * m_b * r * cos(theta) + (-1) * a_2 * m_b * r * sin(theta)) * \
        tanh(500 * r ** (-1) * (2 * theta_dot * r + (-2) * x_dot)) + (-4) * \
        f_d * r * R * (10 * (B_2 + (a_2 ** 2 + c_2 ** 2) * m_b) + c_2 * m_b * \
        r * cos(theta) + (-1) * a_2 * m_b * r * sin(theta)) * tanh(500 * r ** \
        (-1) * ((-2) * theta_dot * r + 2 * x_dot)))

        self.f_4 = lambda x_dot, theta, theta_dot: r ** (-1) * R ** (-1) * (4 * \
        B_2 * J_w + 4 * a_2 ** 2 * J_w * m_b + 4 * c_2 ** 2 * J_w * m_b + 2 *
        B_2 * m_b * r ** 2 + a_2 ** 2 * m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** \
        2 * r ** 2 + 4 * B_2 * m_w * r ** 2 + 4 * a_2 ** 2 * m_b * m_w * r ** \
        2 + 4 * c_2 ** 2 * m_b * m_w * r ** 2 + (a_2 ** 2 + (-1) * c_2 ** 2) * \
        m_b ** 2 * r ** 2 * cos(2 * theta) + 2 * a_2 * c_2 * m_b ** 2 * r ** \
        2 * sin(2 * theta)) ** (-1) * ((-80) * J_w * K ** 2 * theta_dot * r + \
        (-40) * K ** 2 * m_b * theta_dot * r ** 3 + (-80) * K ** 2 * m_w * \
        theta_dot * r ** 3 + (-80) * f_v * J_w * theta_dot * r * R + (-40) * \
        f_v * m_b * theta_dot * r ** 3 * R + (-80) * f_v * m_w * theta_dot * \
        r ** 3 * R + 80 * J_w * K ** 2 * x_dot + 40 * K ** 2 * m_b * r ** 2 * \
        x_dot + 80 * K ** 2 * m_w * r ** 2 * x_dot + 80 * f_v * J_w * R * \
        x_dot + 40 * f_v * m_b * r ** 2 * R * x_dot + 80 * f_v * m_w * r ** 2 \
        * R * x_dot + (-400) * c_2 * K ** 2 * m_b * theta_dot * r ** 2 * \
        cos(theta) + 4 * a_2 * g * J_w * m_b * r * R * cos(theta) + (-400) * \
        c_2 * f_v * m_b * theta_dot * r ** 2 * R * cos(theta) + 2 * a_2 * g * \
        m_b ** 2 * r ** 3 * R * cos(theta) + 4 * a_2 * g * m_b * m_w * r ** 3 \
        * R * cos(theta) + 400 * c_2 * K ** 2 * m_b * r * x_dot * cos(theta) + \
        400 * c_2 * f_v * m_b * r * R * x_dot * cos(theta) + (-2) * a_2 * c_2 \
        * m_b ** 2 * theta_dot ** 2 * r ** 3 * R * cos(2 * theta) + 400 * a_2 \
        * K ** 2 * m_b * theta_dot * r ** 2 * sin(theta) + 4 * c_2 * g * J_w * \
        m_b * r * R * sin(theta) + 400 * a_2 * f_v * m_b * theta_dot * r ** 2 \
        * R * sin(theta) + 2 * c_2 * g * m_b ** 2 * r ** 3 * R * sin(theta) + \
        4 * c_2 * g * m_b * m_w * r ** 3 * R * sin(theta) + (-400) * a_2 * K \
        ** 2 * m_b * r * x_dot * sin(theta) + (-400) * a_2 * f_v * m_b * r * \
        R * x_dot * sin(theta) + a_2 ** 2 * m_b ** 2 * theta_dot ** 2 * r ** 3 \
        * R * sin(2 * theta) + (-1) * c_2 ** 2 * m_b ** 2 * theta_dot ** 2 * r \
        ** 3 * R * sin(2 * theta) + (-2) * f_d * r * R * (2 * J_w + m_b * r ** \
        2 + 2 * m_w * r ** 2 + 10 * c_2 * m_b * r * cos(theta) + (-10) * a_2 * \
        m_b * r * sin(theta)) * tanh(500 * r ** (-1) * (2 * theta_dot * r + \
        (-2) * x_dot)) + 2 * f_d * r * R * (2 * J_w + m_b * r ** 2 + 2 * m_w * \
        r ** 2 + 10 * c_2 * m_b * r * cos(theta) + (-10) * a_2 * m_b * r * \
        sin(theta)) * tanh(500 * r ** (-1) * ((-2) * theta_dot * r + 2 * \
        x_dot)))

        self.g_3 = lambda theta: (-2) * K * r * R ** (-1) * V_nom * ((-10) * \
        (B_2 + (a_2 ** 2 + c_2 ** 2) * m_b) + (-1) * c_2 * m_b * r * \
        cos(theta) + a_2 * m_b * r * sin(theta)) * (2 * B_2 * J_w + 2 * a_2 ** \
        2 * J_w * m_b + 2 * c_2 ** 2 * J_w * m_b + B_2 * m_b * r ** 2 + a_2 ** \
        2 * m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** 2 * r ** 2 + 2 * B_2 * m_w * \
        r ** 2 + 2 * a_2 ** 2 * m_b * m_w * r ** 2 + 2 * c_2 ** 2 * m_b * m_w \
        * r ** 2 + (-1) * c_2 ** 2 * m_b ** 2 * r ** 2 * cos(theta) ** 2 + \
        (-1) * a_2 ** 2 * m_b ** 2 * r ** 2 * sin(theta) ** 2 + a_2 * c_2 * \
        m_b ** 2 * r ** 2 * sin(2 * theta)) ** (-1)

        self.g_4 = lambda theta: (-4) * K * R ** (-1) * V_nom * (2 * J_w + m_b \
        * r ** 2 + 2 * m_w * r ** 2 + 10 * c_2 * m_b * r * cos(theta) + (-10) \
        * a_2 * m_b * r * sin(theta)) * (4 * B_2 * J_w + 4 * a_2 ** 2 * J_w * \
        m_b + 4 * c_2 ** 2 * J_w * m_b + 2 * B_2 * m_b * r ** 2 + a_2 ** 2 * \
        m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** 2 * r ** 2 + 4 * B_2 * m_w * r \
        ** 2 + 4 * a_2 ** 2 * m_b * m_w * r ** 2 + 4 * c_2 ** 2 * m_b * m_w * \
        r ** 2 + (a_2 ** 2 + (-1) * c_2 ** 2) * m_b ** 2 * r ** 2 * cos(2 * \
        theta) + 2 * a_2 * c_2 * m_b ** 2 * r ** 2 * sin(2 * theta)) ** (-1)

    def drift(self, x):
        _, theta, x_dot, theta_dot = x
        return array([x_dot, theta_dot, self.f_3(theta, x_dot, theta_dot), self.f_4(theta, x_dot, theta_dot)])

    def act(self, x):
        _, theta, _, _ = x
        return array([[0], [0], [self.g_3(theta)], [self.g_4(theta)]])

class SegwayOutput(RoboticSystemOutput):
    def __init__(self, segway, ts, theta_ds, theta_dot_ds):
        RoboticSystemOutput.__init__(self, 1)
        self.segway = segway
        theta_ds = array([theta_ds]).T
        theta_dot_ds = array([theta_dot_ds]).T
        self.r, self.r_dot = self.interpolator(ts, theta_ds, theta_dot_ds)
        self.nonzero = array([1, 3])

    def eta(self, x, t):
        return x[self.nonzero] - self.r(t)

    def drift(self, x, t):
        return self.segway.drift(x)[self.nonzero] - self.r_dot(t)

    def decoupling(self, x, t):
        return self.segway.act(x)[self.nonzero]

# Estimated system parameters
m_b_hat = 44.798
m_w_hat = 2.485
J_w_hat = 0.055936595310797
a_2_hat = -0.023227187592750
c_2_hat = 0.166845864363019
B_2_hat = 2.899458828344427
R_hat =  0.086985141514373
K_hat = 0.141344665167821
r_hat = 0.195
g = 9.81
f_d_hat = 0.076067344020759
f_v_hat = 0.002862586216301
V_nom_hat = 57
param_hats = array([m_b_hat, m_w_hat, J_w_hat, a_2_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, f_d_hat, f_v_hat, V_nom_hat])

# True system parameters
delta = 0.2 # Max paramter variation
factors = uniform(1 - delta, 1 + delta, len(param_hats))
params = factors * param_hats
m_b, m_w, J_w, a_2, c_2, B_2, R, K, r, f_d, f_v, V_nom = params

system = SegwaySystem(m_b_hat, m_w_hat, J_w_hat, a_2_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, g, f_d_hat, f_v_hat, V_nom_hat) # Estimated system
system_true = SegwaySystem(m_b, m_w, J_w, a_2, c_2, B_2, R, K, r, g, f_d, f_v, V_nom) # Actual system

# Control design parameters
K_p = array([[0.5]]) # PD controller P gain
K_d = array([[0.1]]) # PD controller D gain
n = 4 # Number of states
m = 1 # Number of inputs
k = 1 # Number of outputs
p = 2 * k # Output vector size
Q = identity(p) # Positive definite Q for CARE

# Simulation parameters
x_0 = array([2, 0, 0, 0]) # Initial condition
dt = 1e-3 # Time step
N = 5000 # Number of time steps
t_eval = dt * arange(N + 1) # Simulation time points

# Loading trajectory data
res = loadmat('./lyapy/trajectories/segway.mat')
t_ds = res['T_d'][:, 0] # Time points
x_ds = res['X_d'] # Desired states
theta_ds = x_ds[:, 1] # Desired angle points
theta_dot_ds = x_ds[:, 3] # Desired angular rate points

# Output and control definitions
output = SegwayOutput(system, t_ds, theta_ds, theta_dot_ds)
pd_controller = PDController(output, K_p, K_d)
lyapunov_function = QuadraticControlLyapunovFunction.build_care(output, Q)
qp_controller = QPController.build_min_norm(lyapunov_function)

# Input to models is state and nonzero component of Lyapunov function gradient
# (sparsity comes from sparsity of system acutation matrix)
input = lambda x, t: concatenate([x, lyapunov_function.grad_V(x, t)[-output.k:]])
s = n + output.k

subsample_rate = 20
width = 0.1
C = 1e3
scaling = 1
offset = 0.1
diff_window = 3
d_hidden = 200
training_loss_threshold = 1e-4
max_epochs = 5000
batch_fraction = 0.1
validation_split = 0.1
num_episodes = 10
weight_final = 0.99
add_episodes = 5
weights = sigmoid_weighting(num_episodes, weight_final, add_episodes)
widths = decay_widths(num_episodes, width, add_episodes)

handler = SimulationHandler(system_true, output, pd_controller, m, lyapunov_function, x_0, t_eval, subsample_rate, input, C, scaling, offset)
trainer = KerasTrainer(input, lyapunov_function, diff_window, subsample_rate, n, s, m, d_hidden, training_loss_threshold, max_epochs, batch_fraction, validation_split)
a, b, train_data, (a_predicts, b_predicts) = trainer.run(handler, weights, widths)
a = evaluator(input, a)
b = evaluator(input, b, scalar_output=True)
aug_controller = QPController.build_aug(pd_controller, m, lyapunov_function, a, b, C)
total_controller = CombinedController([pd_controller, aug_controller], ones(2))

# PD controller simulation
ts, xs = system_true.simulate(x_0, pd_controller, t_eval)
thetas = xs[:, 1]
theta_dots = xs[:, 3]
us = array([pd_controller.u(x, t) for x, t in zip(xs, ts)])

figure()
subplot(2, 1, 1)
plot(ts, thetas, linewidth=2, label='$\\theta$')
plot(ts, theta_dots, linewidth=2, label='$\\dot{\\theta}$')
plot(t_ds, theta_ds, '--', linewidth=2, label='$\\theta_d$')
plot(t_ds, theta_dot_ds, '--', linewidth=2, label='$\\dot{\\theta}_d$')
title('PD Controller')
legend(fontsize=16)
grid()
subplot(2, 1, 2)
plot(ts, us, label='$u$')
legend(fontsize=16)
grid()

# Nominal QP controller simulation
ts, xs = system_true.simulate(x_0, qp_controller, t_eval)
thetas = xs[:, 1]
theta_dots = xs[:, 3]
us = array([qp_controller.u(x, t) for x, t in zip(xs, ts)])

figure()
subplot(2, 1, 1)
plot(ts, thetas, linewidth=2, label='$\\theta$')
plot(ts, theta_dots, linewidth=2, label='$\\dot{\\theta}$')
plot(t_ds, theta_ds, '--', linewidth=2, label='$\\theta_d$')
plot(t_ds, theta_dot_ds, '--', linewidth=2, label='$\\dot{\\theta}_d$')
title('QP Controller')
legend(fontsize=16)
grid()
subplot(2, 1, 2)
plot(ts, us, label='$u$')
legend(fontsize=16)
grid()

# Augmented controller simulation
ts, xs = system_true.simulate(x_0, total_controller, t_eval)
thetas = xs[:, 1]
theta_dots = xs[:, 3]
us = array([total_controller.u(x, t) for x, t in zip(xs, ts)])

figure()
subplot(2, 1, 1)
plot(ts, thetas, linewidth=2, label='$\\theta$')
plot(ts, theta_dots, linewidth=2, label='$\\dot{\\theta}$')
plot(t_ds, theta_ds, '--', linewidth=2, label='$\\theta_d$')
plot(t_ds, theta_dot_ds, '--', linewidth=2, label='$\\dot{\\theta}_d$')
title('Augmented Controller')
legend(fontsize=16)
grid()
subplot(2, 1, 2)
plot(ts, us, label='$u$')
legend(fontsize=16)
grid()

# Additional plots
output_true = SegwayOutput(system_true, t_ds, theta_ds, theta_dot_ds)
lyapunov_function_true = QuadraticControlLyapunovFunction.build_care(output_true, Q)
a_tests = array([a(x, t) for x, t in zip(xs, ts)])
a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
b_tests = array([b(x, t) for x, t in zip(xs, ts)])
b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])

figure()
suptitle('Test data', fontsize=16)
subplot(2, 1, 1)
plot(a_tests, label='$\\widehat{a}$', linewidth=2)
plot(a_trues, label='$a$', linewidth=2)
legend(fontsize=16)
grid()
subplot(2, 1, 2)
plot(b_tests, label='$\\widehat{b}$', linewidth=2)
plot(b_trues, label='$b$', linewidth=2)
legend(fontsize=16)
grid()

xs, ts, _, _, u_noms, u_perts, V_dot_rs = train_data
thetas = xs[:, 1]
theta_dots = xs[:, 3]
a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])
V_dot_r_trues = array([lyapunov_function_true.V_dot(x, u_nom + u_pert, t) - lyapunov_function.V_dot(x, u_nom, t) for x, u_nom, u_pert, t in zip(xs, u_noms, u_perts, ts)])

figure()
suptitle('Episode data', fontsize=16)
subplot(2, 1, 1)
title('State data', fontsize=16)
plot(thetas, linewidth=2, label='$\\theta$')
plot(theta_dots, linewidth=2, label='$\\dot{\\theta}$')
legend(fontsize=16)
grid()
subplot(2, 1, 2)
title('Control data', fontsize=16)
plot(u_noms + u_perts, label='$u$', linewidth=2)
legend(fontsize=16)
grid()

figure()
plot(V_dot_rs, linewidth=2, label='$\\widehat{\\dot{V}}_r$')
plot(V_dot_r_trues, '--', linewidth=2, label='$\\dot{V}_r$')
title('Numerical differentiation', fontsize=16)
legend(fontsize=16)
grid()

figure()
suptitle('Training results', fontsize=16)
subplot(2, 1, 1)
plot(a_predicts, label='$\\widehat{a}$', linewidth=2)
plot(a_trues, '--', label='$a$', linewidth=2)
legend(fontsize=16)
grid()
subplot(2, 1, 2)
plot(b_predicts, label='$\\widehat{b}$', linewidth=2)
plot(b_trues, '--', label='$b$', linewidth=2)
legend(fontsize=16)
grid()

show()
