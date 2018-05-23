from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from matplotlib.pyplot import figure, grid, legend, plot, scatter, show, xlabel, ylabel
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, concatenate, linspace, sqrt, pi
from numpy.random import rand, randn

from ..controllers import PendulumController
from ..learning import connect_models, differentiator
from ..systems import Pendulum

m, g, l = 0.25, 9.81, 0.5
pendulum = Pendulum(m, g, l)
m_hat, K = m * 0.8, array([1, 2])
pendulum_controller = PendulumController(pendulum, m_hat, K)
u, V, dV = pendulum_controller.synthesize()

x_0 = array([0.1, 0])
t_eval = linspace(0, 10, 1e3)
ts, xs = pendulum.simulate(u, x_0, t_eval)

us = array([u(x, t) for x, t in zip(xs, ts)])
Vs = array([V(x, t) for x, t in zip(xs, ts)])
dVs = array([dV(x, u, t) for x, u, t in zip(xs, us, ts)])

figure()
plot(ts, xs, linewidth=3)
grid()
legend(['$\\theta$', '$\\dot{\\theta}$'])

figure()
plot(ts, us, linewidth=3)
grid()
legend(['$u$'])

figure()
plot(ts, Vs, linewidth=3)
grid()
legend(['$V$'])

figure()
plot(ts, dVs, linewidth=3)
grid()
legend(['$\\dot{V}$'])

N, dt = 1000, 1e-2
diff = differentiator(2, dt)
x_0s = array([(rand(N) - 0.5) * pi, sqrt(2) * randn(N)]).T
t_eval = array([0, dt])
sols = [pendulum.simulate(u, x_0, t_eval) for x_0 in x_0s]

xs = array([xs[-1] for _, xs in sols])
ts = array([ts[-1] for ts, _ in sols])
us = array([u(x, t) for x, t in zip(xs, ts)])
Vs = [array([V(x, t) for x, t in zip(xs, ts)]) for ts, xs in sols]
dV_hats = concatenate([diff(Vs) for Vs in Vs])

figure()
scatter(xs[:, 0], xs[:, 1])
xlabel('$\\theta$', fontsize=16)
ylabel('$\\dot{\\theta}$', fontsize=16)
grid()

k = 200
reg = l2(1e-3)

w = Sequential()
w.add(Dense(k, input_shape=(2,), activation='relu', kernel_regularizer=reg))
w.add(Dense(1, kernel_regularizer=reg))

b = Sequential()
b.add(Dense(k, input_shape=(2,), activation='relu', kernel_regularizer=reg))
b.add(Dense(1, kernel_regularizer=reg))

model = connect_models(w, b)
model.compile('sgd', 'mean_squared_error')
model.fit([xs, us], dV_hats, batch_size=N // 100, epochs=200, validation_split=0.2)

t = 0
N = 25
thetas = linspace(-1, 1, N)
theta_dots = linspace(-2, 2, N)
xs = array([array([theta, theta_dot]) for theta in thetas for theta_dot in theta_dots])
us = array([u(x, t) for x in xs])
dVs = array([dV(x, u, t) for x, u in zip(xs, us)])
dV_hats = model.predict([xs, us])[:, 0]

pendulum_controller = PendulumController(pendulum, m, K)
_, _, dV_actual = pendulum_controller.synthesize()
dV_actuals = array([dV_actual(x, u, t) for x, u in zip(xs, us)])

f = figure()
ax = f.add_subplot(221, projection='3d')
ax.plot_trisurf(xs[:, 0], xs[:, 1], dVs)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$\\dot{V}_d$', fontsize=16)

ax = f.add_subplot(222, projection='3d')
ax.plot_trisurf(xs[:, 0], xs[:, 1], dV_actuals)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$\\dot{V}$', fontsize=16)

ax = f.add_subplot(223, projection='3d')
ax.plot_trisurf(xs[:, 0], xs[:, 1], dV_hats)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$\\hat{\\dot{V}}$', fontsize=16)

ax = f.add_subplot(224, projection='3d')
ax.plot_trisurf(xs[:, 0], xs[:, 1], dV_actuals - dV_hats)
ax.set_xlabel('$\\theta$', fontsize=16)
ax.set_ylabel('$\\dot{\\theta}$', fontsize=16)
ax.set_title('$\\dot{V} - \\hat{\\dot{V}}$', fontsize=16)

show()
