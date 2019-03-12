"""Planar segway example."""

from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import arange, array, concatenate, cos, identity, linspace, ones, sin, tanh, tile, zeros
from numpy.random import uniform
from scipy.io import loadmat, savemat
from sys import argv

from ..controllers import CombinedController, PDController, QPController, SaturationController
from ..learning import evaluator, KerasTrainer, sigmoid_weighting, SimulationHandler
from ..lyapunov_functions import RESQuadraticControlLyapunovFunction
from ..outputs import PDOutput, RoboticSystemOutput
from ..systems import AffineControlSystem

filename = argv[1]

class SegwaySystem(AffineControlSystem):
    """Planar Segway system. State is [x, theta, x_dot, theta_dot], where x is
    the position of the Segway base in m, x_dot is the velocity in m / sec,
    theta is the angle of the frame in rad clockwise from upright, and
    theta_dot is the angular rate in rad / sec. The input is [u], where u is
    positive or negative percent of maximum motor voltage.

    Attributes:
    x_ddot drift component, f3: float * float * float -> float
    theta_ddot drift component, f4: float * float * float -> float
    x_ddot actuation component, g3: float -> float
    theta_ddot actuation component, g4: float -> float
    """

    def __init__(self, m_b=44.798, m_w=2.485, J_w=0.055936595310797, a_2=-0.023227187592750, c_2=0.166845864363019, B_2=2.899458828344427, R= 0.086985141514373, K=0.141344665167821, r=0.195, g=9.81, f_d=0.076067344020759, f_v=0.002862586216301, V_nom=57):
        """Initialize a SegwaySystem object.

        Inputs:
        Mass of frame (kg), m_b: float
        Mass of one wheel (kg), m_w: float
        Inertia of wheel (kg*m^2), J_w: float
        x position of frame (m), a_2: float
        z position of frame (m), c_2: float
        yy inertia of frame (kg*m^2), B_2: float
        Electrical resistance of motors (Ohm), R: float
        Torque constant of motors (N*m/A), K: float
        Radius of wheels (m), r: float
        Gravity constant (m/s^2), g: float
        Dry friction coefficient (N*m), f_d: float
        Viscous friction coefficient (N*m*s), f_v: float
        Nominal battery voltage (V), V_nom: float
        """

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
        return array([x_dot, theta_dot, self.f_3(x_dot, theta, theta_dot), self.f_4(x_dot, theta, theta_dot)])

    def act(self, x):
        _, theta, _, _ = x
        return array([[0], [0], [self.g_3(theta)], [self.g_4(theta)]])

class SegwayOutput(RoboticSystemOutput):
    """Segway output for trajectory tracking.

	Outputs are eta = (theta - theta_d(t), theta_dot - theta_dot_d(t)), where
	theta and theta_dot are as specified in the Segway class. theta_d and
    theta_dot_d are interpolated from a sequence of desired angles, theta_ds,
    and angle rates, theta_dot_ds.

	Attributes:
    List of relative degrees, vector_relative_degree: int list
    Permutation indices, permutation_idxs: numpy array (2,)
    Reverse permutation indices, reverse_permutation_idxs: numpy array (2,)
    Indices of k outputs when eta in block form, relative_degree_idxs: numpy array (1,)
    Indices of permutation into form with highest order derivatives in block, blocking_idxs: numpy array (2,)
    Indices of reverse permutation into form with highest order derivatives in block, unblocking_idxs: numpy array (2,)
    Linear output update matrix after decoupling inversion and drift removal, F: numpy array (2, 2)
    Linear output actuation matrix after decoupling inversion and drift removal, G: numpy array (2, 1)
    """

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

class SegwayEqOutput(PDOutput):
    """Segway output for upright equilibrium tracking.

    Outputs are (x, theta, x_dot, theta_dot), as in the Segway class.
    """

    def __init__(self, x_eq):
        """Initialize a SegwayEqOutput object.

        Inputs:
        Equilibrium state, x_eq: numpy array (4,)
        """

        self.x_eq = x_eq

    def eta(self, x, t):
        return x - x_eq

    def proportional(self, x, t):
        return self.eta(x, t)[:2]

    def derivative(self, x, t):
        return self.eta(x, t)[2:]

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

# True system parameters
m_b = 46.31206379797803
m_w = 2.633437914482628
J_w = 0.0568084550600686
a_2 = -0.024976309468173894
c_2 = 0.17247676784552285
B_2 = 3.1290944695185563
R = 0.08095373131599383
K = 0.14276458237675732
r = 0.20383506582477764
f_d = 0.07201659972874044
f_v = 0.00308764872345703
V_nom = 54.73910741841314

system = SegwaySystem(m_b_hat, m_w_hat, J_w_hat, a_2_hat, c_2_hat, B_2_hat, R_hat, K_hat, r_hat, g, f_d_hat, f_v_hat, V_nom_hat) # Estimated system
system_true = SegwaySystem(m_b, m_w, J_w, a_2, c_2, B_2, R, K, r, g, f_d, f_v, V_nom) # Actual system

# Control design parameters
K_p = array([[3.2]]) # PD controller P gain
K_d = array([[0.1]]) # PD controller D gain
n = 4 # Number of states
m = 1 # Number of inputs
k = 1 # Number of outputs
p = 2 * k # Output vector size
Q = identity(p) # Positive definite Q for CARE
epsilon = .1 # Convergence rate scale factor for RES-CLF
lower_bounds = array([-1]) # Voltage percentage lower bound
upper_bounds = array([1]) # Voltage percentage upper bound

# Simulation parameters
x_0 = array([2, 0, 0, 0]) # Initial condition
dt = 1.0e-3 # Time step
N = 5000 # Number of time steps
t_eval = dt * arange(N + 1) # Simulation time points

# Loading trajectory data
res = loadmat('./lyapy/trajectories/segway.mat')
t_ds = res['T_d'][:, 0] # Time points
x_ds = res['X_d'] # Desired states
theta_ds = x_ds[:, 1] # Desired angle points
theta_dot_ds = x_ds[:, 3] # Desired angular rate points

H = array([[10]]) # Smoothing cost

# Output and control definitions
output = SegwayOutput(system, t_ds, theta_ds, theta_dot_ds)
pd_controller = PDController(output, K_p, K_d)
lyapunov_function = RESQuadraticControlLyapunovFunction.build_care(output, Q, epsilon)
qp_controller = QPController.build_min_norm(lyapunov_function, H=H)
qp_controller = SaturationController(output, qp_controller, m, lower_bounds, upper_bounds)

# Input to models is state and nonzero component of Lyapunov function gradient
# (sparsity comes from sparsity of system acutation matrix)
input = lambda x, t: concatenate([x, lyapunov_function.grad_V(x, t)[-output.k:]])
s = n + output.k

subsample_rate = 20
width = 0.25
C = 1e0
scaling = 1
offset = 0
diff_window = 3
d_hidden = 2000
N_hidden = 1
training_loss_threshold = 1e-3
max_epochs = 5000
batch_fraction = 0.1
validation_split = 0.1
num_episodes = 20
weight_final = 0.99
add_episodes = 0

weights = sigmoid_weighting(num_episodes, weight_final)
weights = concatenate([zeros(add_episodes), weights, ones(add_episodes)])
widths = concatenate([width * ones(num_episodes // 2 + add_episodes), linspace(width, 0, num_episodes // 2 + add_episodes)])

handler = SimulationHandler(system_true, output, pd_controller, m, lyapunov_function, x_0, t_eval, subsample_rate, input, C, H, scaling, offset, lower_bounds, upper_bounds)
trainer = KerasTrainer(input, lyapunov_function, diff_window, subsample_rate, n, s, m, d_hidden, N_hidden, training_loss_threshold, max_epochs, batch_fraction, validation_split)
a, b, train_data, log = trainer.run(handler, weights, widths)
(a_predicts, b_predicts, deltas), (a_logs, b_logs) = log
a = evaluator(input, a)
b = evaluator(input, b, scalar_output=True)
aug_controller = QPController.build_aug(pd_controller, m, lyapunov_function, a, b, C, H=H)
total_controller = CombinedController([pd_controller, aug_controller], ones(2))
total_controller = SaturationController(output, total_controller, m, lower_bounds, upper_bounds)

# PD controller simulation
pd_controller_sat = SaturationController(output, pd_controller, m, lower_bounds, upper_bounds)
ts, xs = system_true.simulate(x_0, pd_controller_sat, t_eval)
savemat('./output/segway_pd_data.mat', {'xs': xs, 'ts': ts})
thetas = xs[:, 1]
theta_dots = xs[:, 3]
us = pd_controller_sat.evaluate(xs, ts)

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

# EQ tracking output and control definitions
x_eq = array([0, 0.1383, 0, 0])
K_p = array([[0, 0.8]])
K_d = array([[0.5, 0.3]])
eq_output = SegwayEqOutput(x_eq)
eq_controller = PDController(eq_output, K_p, K_d)
eq_controller = SaturationController(output, eq_controller, m, lower_bounds, upper_bounds)
ts, xs = system_true.simulate(xs[-1], eq_controller, t_eval)
savemat('./output/segway_eq_data.mat', {'xs': xs, 'ts': ts})

# Nominal QP controller simulation
ts, xs = system_true.simulate(x_0, qp_controller, t_eval)
savemat('./output/segway_qp_data.mat', {'xs': xs, 'ts': ts})
thetas = xs[:, 1]
theta_dots = xs[:, 3]
us = qp_controller.evaluate(xs, ts)

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
savemat('./output/segway_aug_data_' + filename + '.mat', {'xs': xs, 'ts': ts})
us = total_controller.evaluate(xs, ts)
delta_tests = aug_controller.evaluate_slack(xs, ts)

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
lyapunov_function_true = RESQuadraticControlLyapunovFunction.build_care(output_true, Q, epsilon)

x_logs = zeros((0, n))
u_logs = zeros((0, m))
t_logs = zeros(0)
a_log_predicts = zeros((0, m))
b_log_predicts = zeros(0)

print('POST PROCESSING...')

for episode, (a_log, b_log, weight) in enumerate(zip(a_logs, b_logs, weights)):
    print('EPISODE', episode)

    a_log = evaluator(input, a_log)
    b_log = evaluator(input, b_log, scalar_output=True)
    aug_controller = QPController.build_aug(pd_controller, m, lyapunov_function, a_log, b_log, C, H)
    total_controller = CombinedController([pd_controller, aug_controller], array([1, weight]), output)
    total_controller = SaturationController(output, total_controller, m, lower_bounds, upper_bounds)
    t_log, x_log = system_true.simulate(x_0, total_controller, t_eval)
    u_log = total_controller.evaluate(x_log, t_log)
    x_logs = concatenate([x_logs, x_log])
    u_logs = concatenate([u_logs, u_log])
    t_logs = concatenate([t_logs, t_log])

    a_log_predict = array([a_log(x, t) for x, t in zip(xs, ts)])
    b_log_predict = array([b_log(x, t) for x, t in zip(xs, ts)])
    a_log_predicts = concatenate([a_log_predicts, a_log_predict])
    b_log_predicts = concatenate([b_log_predicts, b_log_predict])

a_tests = array([a(x, t) for x, t in zip(xs, ts)])
a_trues = array([lyapunov_function_true.decoupling(x, t) - lyapunov_function.decoupling(x, t) for x, t in zip(xs, ts)])
b_tests = array([b(x, t) for x, t in zip(xs, ts)])
b_trues = array([lyapunov_function_true.drift(x, t) - lyapunov_function.drift(x, t) for x, t in zip(xs, ts)])

x_d_logs = array([output.r(t) for t in ts])
x_d_logs = tile(x_d_logs, [num_episodes + 2 * add_episodes, 1])
t_d_logs = t_logs
a_log_trues = tile(a_trues, [num_episodes + 2 * add_episodes, 1])
b_log_trues = tile(b_trues, num_episodes + 2 * add_episodes)

print(len(x_logs), len(x_d_logs))

savemat('./output/segway_eval_data_' + filename + '.mat', {'xs': x_logs, 'us': u_logs, 'ts': t_logs, 'a_hats': a_log_predicts, 'b_hats': b_log_predicts, 'a_trues': a_log_trues, 'b_trues': b_log_trues, 'a_tests': a_tests, 'b_tests': b_tests})

figure()
title('Model evaluation', fontsize=16)
plot(x_logs[:, 1], linewidth=2, label='$\\theta$')
plot(x_d_logs[:, 0], linewidth=2, label='$\\theta_d$')
grid()
legend(fontsize=16)

figure()
suptitle('Model evaluation', fontsize=16)
subplot(2, 1, 1)
plot(a_log_predicts, linewidth=2, label='$\\hat{a}$')
plot(a_log_trues, linewidth=2, label='$a$')
grid()
legend(fontsize=16)
subplot(2, 1, 2)
plot(b_log_predicts, linewidth=2, label='$\\hat{b}$')
plot(b_log_trues, linewidth=2, label='$b$')
grid()
legend(fontsize=16)

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

savemat('./output/segway_full_data.mat', {'xs': xs, 'ts': ts})

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
