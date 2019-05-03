from numpy import arange, argsort, array, concatenate, cumsum, diag, diff, dot, ones, real, reshape, zeros
from numpy.linalg import eig, solve, norm
from numpy.random import multivariate_normal, randn
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, solve_continuous_are, solve_continuous_lyapunov

class Dynamics:
    def eval(self, x, t):
        pass

    def eval_dot(self, x, u, t):
        pass

class SystemDynamics(Dynamics):
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def eval(self, x, t):
        return x

    def simulate(self, x_0, controller, ts, atol=1e-6, rtol=1e-6):
        assert len(x_0) == self.n

        N = len(ts)
        xs = zeros((N, self.n))
        us = [None] * (N - 1)

        xs[0] = x_0
        for j in range(N - 1):
            x = xs[j]
            t = ts[j]
            u = controller.eval(x, t)
            us[j] = u
            u = controller.process(u)
            x_dot = lambda t, x: self.eval_dot(x, u, t)
            t_span = [t, ts[j + 1]]
            res = solve_ivp(x_dot, t_span, x, atol=atol, rtol=rtol)
            xs[j + 1] = res.y[:, -1]

        return xs, us

class AffineDynamics(Dynamics):
    def drift(self, x, t):
        pass

    def act(self, x, t):
        pass

    def eval_dot(self, x, u, t):
        return self.drift(x, t) + dot(self.act(x, t), u)

class LinearizableDynamics(Dynamics):
    def linear_system(self):
        pass

    def closed_loop_linear_system(self, K):
        A, B = self.linear_system()
        return A - dot(B, K)

class LinearSystemDynamics(SystemDynamics, AffineDynamics, LinearizableDynamics):
    def __init__(self, A, B):
        n, m = B.shape
        assert A.shape == (n, n)

        SystemDynamics.__init__(self, n, m)
        self.A = A
        self.B = B

    def drift(self, x, t):
        return dot(self.A, x)

    def act(self, x, t):
        return self.B

    def linear_system(self):
        return self.A, self.B

class PDDynamics(Dynamics):
    def proportional(self, x, t):
        pass

    def derivative(self, x, t):
        pass

class FBLinDynamics(AffineDynamics, LinearizableDynamics):
    def __init__(self, relative_degrees, perm=None):
        self.relative_degrees = relative_degrees
        self.relative_degree_idxs = cumsum(relative_degrees) - 1
        if perm is None:
            perm = arange(sum(relative_degrees))
        self.perm = perm
        self.inv_perm = argsort(perm)

    def select(self, arr):
        return arr[self.relative_degree_idxs]

    def permute(self, arr):
        return arr[self.perm]

    def inv_permute(self, arr):
        return arr[self.inv_perm]

    def linear_system(self):
        F = block_diag(*[diag(ones(gamma - 1), 1) for gamma in self.relative_degrees])
        G = (identity(sum(self.relative_degrees))[self.relative_degree_idxs]).T

        F = (self.inv_permute((self.inv_permute(F)).T)).T
        G = self.inv_permute(G)

        return F, G

class RoboticDynamics(FBLinDynamics, PDDynamics):
    def __init__(self, k):
        relative_degrees = [2] * k
        perm = concatenate([array([j, j + k]) for j in range(k)])
        FBLinDynamics.__init__(self, relative_degrees, perm)
        self.k = k

    def proportional(self, x, t):
        return self.eval(x, t)[:self.k]

    def derivative(self, x, t):
        return self.eval(x, t)[self.k:]

class Controller:
    def __init__(self, dynamics):
        self.dynamics = dynamics

    def eval(self, x, t):
        pass

    def process(self, u):
        return u

class LinearController(Controller):
    def __init__(self, affine_dynamics, K):
        Controller.__init__(self, affine_dynamics)
        self.K = K

    def eval(self, x, t):
        return -dot(self.K, self.dynamics.eval(x, t))

class PDController(Controller):
    def __init__(self, pd_dynamics, K_p, K_d):
        Controller.__init__(self, pd_dynamics)
        self.K_p = K_p
        self.K_d = K_d

    def eval(self, x, t):
        e_p = dynamics.proportional(x, t)
        e_d = dynamics.derivative(x, t)
        return -dot(self.K_p, e_p) - dot(self.K_d, e_d)

class FBLinController(Controller):
    def __init__(self, fb_lin_dynamics, K):
        Controller.__init__(self, fb_lin_dynamics)
        self.linear_controller = LinearController(fb_lin_dynamics, K)
        self.select = fb_lin_dynamics.select
        self.permute = fb_lin_dynamics.permute

    def eval(self, x, t):
        drift = self.select(self.permute(self.dynamics.drift(x, t)))
        act = self.select(self.permute(self.dynamics.act(x, t)))
        return solve(act, -drift + self.linear_controller.eval(x, t))

class RandomController(Controller):
    def __init__(self, controller, cov, reps=2):
        Controller.__init__(self, controller.dynamics)
        self.controller = controller
        self.cov = cov
        self.m, _ = cov.shape
        self.reps = reps
        self.counter = reps
        self.pert = self.sample()

    def sample(self):
        return multivariate_normal(zeros(self.m), self.cov)

    def eval(self, x, t):
        if self.counter == 0:
            self.pert = self.sample()
            self.counter = self.reps + 1
        self.counter = self.counter - 1
        return self.controller.eval(x, t), self.pert

    def process(self, u):
        u_nom, u_pert = u
        return u_nom + u_pert

class QuadraticCLF(Dynamics):
    def __init__(self, dynamics, P):
        self.dynamics = dynamics
        self.P = P

    def eval(self, x, t):
        z = self.dynamics.eval(x, t)
        return dot(z, dot(self.P, z))

    def eval_grad(self, x, t):
        z = self.dynamics.eval(x, t)
        return 2 * dot(self.P, z)

    def eval_dot(self, x, u, t):
        return dot(self.eval_grad(x, t), self.dynamics.eval_dot(x, u, t))

class AffineQuadCLF(AffineDynamics, QuadraticCLF):
    def __init__(self, affine_dynamics, P):
        QuadraticCLF.__init__(self, affine_dynamics, P)

    def drift(self, x, t):
        return dot(self.eval_grad(x, t), self.dynamics.drift(x, t))

    def act(self, x, t):
        return dot(self.eval_grad(x, t), self.dynamics.act(x, t))

    def build_care(affine_linearizable_dynamics, Q, R):
        F, G = affine_linearizable_dynamics.linear_system()
        P = solve_continuous_are(F, G, Q, R)
        return AffineQuadCLF(affine_linearizable_dynamics, P)

    def build_ctle(affine_linearizable_dynamics, Q, K):
        A = affine_linearizable_dynamics.closed_loop_linear_system(K)
        P = solve_continuous_lyapunov(A.T, -Q)
        return AffineQuadCLF(affine_linearizable_dynamics, P)

class LQRController(Controller):
    def __init__(self, affine_linearizable_dynamics, P, R):
        Controller.__init__(self, affine_linearizable_dynamics)
        self.P = P
        self.R = R

    def eval(self, x, t):
        _, B = self.dynamics.linear_system()
        return -solve(self.R, dot(B.T, dot(self.P, self.dynamics.eval(x, t)))) / 2

    def build(affine_linearizable_dynamics, Q, R):
        lyap = AffineQuadCLF.build_care(affine_linearizable_dynamics, Q, R)
        return LQRController(affine_linearizable_dynamics, lyap.P, R)

def rand_orthogonal(n):
    M = randn(n, n)
    M = dot(M.T, M)
    _, Q = eig(M)
    return Q

def differentiate(xs, ts, L=3):
    half_L = (L - 1) // 2
    b = zeros(L)
    b[1] = 1

    def diff(xs, ts):
        t_0 = ts[half_L]
        t_diffs = reshape(ts - t_0, (L, 1))
        pows = reshape(arange(L), (1, L))
        A = (t_diffs ** pows).T
        w = solve(A, b)
        return dot(w, xs)

    return array([diff(xs[k - half_L:k + half_L + 1], ts[k - half_L:k + half_L + 1]) for k in range(half_L, len(ts) - half_L)])

def lqr_cost(Q, R, xs, us, ts):
    stage_costs = array([dot(x, dot(Q, x)) + dot(u, dot(R, u)) for x, u in zip(xs, us)])
    return cumsum(stage_costs * diff(ts))

def matrix_select(A, idxs):
    return (((A[idxs]).T)[idxs]).T

def compare_unstable(A, A_hat):
    d, U = eig(A)
    idxs = real(d) >= 0
    D = diag(d)

    A_unstable = matrix_select(D, idxs)
    A_hat_unstable = matrix_select(solve(U, dot(A_hat, U)), idxs)

    return norm(A_unstable - A_hat_unstable)
