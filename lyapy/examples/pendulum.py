from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from matplotlib.pyplot import figure, grid, legend, plot, scatter, show, subplot, suptitle, xlabel, ylabel
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, concatenate, linspace, pi, sqrt
from numpy.random import rand, randn

from ..controllers import PendulumController
from ..learning import augmenting_controller, connect_models, constant_controller, differentiator, sum_controller
from ..systems import Pendulum

m, g, l = 0.25, 9.81, 0.5
pendulum = Pendulum(m, g, l)
m_hat, K = m * 0.8, array([1, 2])
pendulum_controller = PendulumController(pendulum, m_hat, K)
u, V, dV = pendulum_controller.synthesize()
pendulum_controller_true = PendulumController(pendulum, m, K)
_, _, dV_true = pendulum_controller_true.synthesize()

x_0 = array([1, 0])
t_eval = linspace(0, 10, 1e3)
ts, xs = pendulum.simulate(u, x_0, t_eval)

us = array([u(x, t) for x, t in zip(xs, ts)])
Vs = array([V(x, t) for x, t in zip(xs, ts)])
dV_trues = array([dV_true(x, u, t) for x, u, t in zip(xs, us, ts)])

figure()
subplot(2, 2, 1)
plot(ts, xs, linewidth=3)
grid()
legend(['$\\theta$', '$\\dot{\\theta}$'])
subplot(2, 2, 2)
plot(ts, us, linewidth=3)
grid()
legend(['$u$'])
subplot(2, 2, 3)
plot(ts, Vs, linewidth=3)
grid()
legend(['$V$'])
subplot(2, 2, 4)
plot(ts, dV_trues, linewidth=3)
grid()
legend(['$\\dot{V}$'])
suptitle('Nominal control system', fontsize=16)

k, reg = 100, l2(1e-3)
w = Sequential()
w.add(Dense(k, input_shape=(2,), kernel_regularizer=reg, activation='relu'))
w.add(Dense(1, kernel_regularizer=reg))
b = Sequential()
b.add(Dense(k, input_shape=(2,), kernel_regularizer=reg, activation='relu'))
b.add(Dense(1, kernel_regularizer=reg))
model = connect_models(w, b)

eta, final_eta, epochs = 4e-2, 4e-3, 1000
decay = 1 - (final_eta / eta) ** (1 / epochs)
model.compile(SGD(lr=eta, decay=decay), 'mean_squared_error')

N, sigma, dt = 2000, sqrt(0.1), 1e-2
diff = differentiator(2, dt)

u_0s = [constant_controller(u_0) for u_0 in sigma * randn(N, 1)]
us = [sum_controller([u, u_0]) for u_0 in u_0s]
x_0s = array([(rand(N) - 0.5) * pi, randn(N)]).T
t_eval = array([0, dt])
sols = [pendulum.simulate(u, x_0, t_eval) for u, x_0 in zip(us, x_0s)]

xs = array([xs[-1] for _, xs in sols])
ts = array([ts[-1] for ts, _ in sols])
us = array([u(x, t) for x, u, t in zip(xs, us, ts)])
dV_hats = concatenate([diff(array([V(x, t) for x, t in zip(xs, ts)])) for ts, xs in sols])

figure()
suptitle('State space samples', fontsize=16)
scatter(xs[:, 0], xs[:, 1])
grid()
xlabel('$\\theta$', fontsize=16)
ylabel('$\\dot{\\theta}$', fontsize=16)

model.fit([xs, us], dV_hats, epochs=epochs, batch_size=N // 10, validation_split=0.2)

N = 20
thetas = linspace(-1, 1, N)
theta_dots = linspace(-1, 1, N)
xs = array([array([theta, theta_dot]) for theta in thetas for theta_dot in theta_dots])
LfVs = array([pendulum_controller_true.LfV(x) for x in xs])
LgVs = concatenate([pendulum_controller_true.LgV(x) for x in xs])
ws = w.predict(xs)[:, 0]
bs = b.predict(xs)[:, 0]

f = figure()
ax = f.add_subplot(221, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], LfVs)
ax.scatter(xs[:, 0], xs[:, 1], bs)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.legend(['$L_fV(x)$', '$b(x)$'], fontsize=16)

ax = f.add_subplot(222, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], LfVs - bs)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$L_fV(x) - b(x)$', fontsize=16)

ax = f.add_subplot(223, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], LgVs)
ax.scatter(xs[:, 0], xs[:, 1], ws)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.legend(['$L_gV(x)$', '$w(x)$'], fontsize=16)

ax = f.add_subplot(224, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], LgVs - ws)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$L_gV(x) - w(x)$', fontsize=16)
suptitle('$\\dot{V}$ estimation', fontsize=16)

t = 0

us = array([u(x, t) for x in xs])
dVs = array([dV(x, u, t) for x, u in zip(xs, us)])

C = 1e3
u_aug = augmenting_controller(u, w, b, dV, C)
u_aug = sum_controller([u, u_aug])
u_augs = array([u_aug(x, t) for x in xs])
dV_trues = array([dV_true(x, u_aug, t) for x, u_aug in zip(xs, u_augs)])
dV_noms = array([dV_true(x, u, t) for x, u in zip(xs, us)])

f = figure()
ax = f.add_subplot(121, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], us[:, 0])
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.legend(['$u$'], fontsize=16)

ax = f.add_subplot(122, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], u_augs[:, 0])
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.legend(['$u_{aug}$'], fontsize=16)
suptitle('Nominal vs Augmented controllers', fontsize=16)

f = figure()
ax = f.add_subplot(121, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], dV_trues)
ax.scatter(xs[:, 0], xs[:, 1], dVs)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.legend(['$\\dot{V}$', '$\\dot{V}_d$'], fontsize=16)

ax = f.add_subplot(122, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], dV_trues - dVs)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$\\dot{V}$ - $\\dot{V}_d$', fontsize=16)
suptitle('Error in $\\dot{V}$ estimation', fontsize=16)

f = figure()
ax = f.add_subplot(121, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], dV_noms)
ax.scatter(xs[:, 0], xs[:, 1], dV_trues)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.legend(['$\\dot{V}$', '$\\dot{V}_{aug}$'], fontsize=16)

ax = f.add_subplot(122, projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], dV_noms - dV_trues)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$\\dot{V} - \\dot{V}_{aug}$', fontsize=16)
suptitle('Difference in $\\dot{V}$ with augmentation', fontsize=16)

x_0 = array([1, 0])
t_eval = linspace(0, 10, 1e3)
ts, xs = pendulum.simulate(u_aug, x_0, t_eval)

u_augs = array([u_aug(x, t) for x, t in zip(xs, ts)])
Vs = array([V(x, t) for x, t in zip(xs, ts)])
dV_trues = array([dV_true(x, u_aug, t) for x, u_aug, t in zip(xs, u_augs, ts)])

figure()
subplot(2, 2, 1)
plot(ts, xs, linewidth=3)
grid()
legend(['$\\theta$', '$\\dot{\\theta}$'])
subplot(2, 2, 2)
plot(ts, u_augs, linewidth=3)
grid()
legend(['$u$'])
subplot(2, 2, 3)
plot(ts, Vs, linewidth=3)
grid()
legend(['$V$'])
subplot(2, 2, 4)
plot(ts, dV_trues, linewidth=3)
grid()
legend(['$\\dot{V}$'])
suptitle('Augmented control system', fontsize=16)

show()
