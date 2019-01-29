# LyaPy

Library for simulation of nonlinear control systems, control design, and Lyapunov-based learning.

## Installation and usage

You will need Python3 and PIP package manager. Clone the repository in a directory of your choice, we will refer to it as `DIR`.

### macOS

Navigate to the directory `DIR`. Create a virtual environment with target directory `.venv` with

`python3 -m venv .venv`

To activate the virtual environment, use

`source .venv/bin/activate`

When you want to deactivate the environment, use

`deactivate`

To install all dependencies, use

`pip3 install -r requirements.txt`

To run an example, use

`python3 -m lyapy.examples.inverted_pendulum`

## Notation

Let `x` denote a state vector, `t` denote a time, `eta` denote an output, `x_dot` = `dx/dt` denote a state derivative, and `u` denote a control input.

## Systems

System classes are the fundamental classes used to simulate dynamical systems.

### `System`

Abstract class for simulating continuous-time dynamics of the form `x_dot = f(t, x)`.

### `ControlSystem`

`System` -> `ControlSystem`

Abstract class for simulating control systems of the form `x_dot = f(x, u, t)`. `u` is computed by a `Controller` object only at specified time steps.

### `AffineControlSystem`

`System` -> `ControlSystem` -> `AffineControlSystem`

Abstract class for simulating affine control systems of the form `x_dot = f(x) + g(x) * u`. As with a `ControlSystem`, `u` is computed by a `Controller` object only at specified time steps.

## Outputs

Output classes define control objectives as functions of state and time. They are used to specify controllers and Lyapunov functions.

### `Output`

Abstract class for evaluating control objectives of the form `eta(x, t)`.

### `AffineDynamicOutput`

`Output` -> `AffineDynamicOutput`

Abstract class for evaluating differentiable control objectives with dynamics `eta_dot` that decompose as `eta_dot = drift(x, t) + decoupling(x, t) * u`.

### `FeedbackLinearizableOutput`

`Output` -> `AffineDynamicOutput` -> `FeedbackLinearizableOutput`

Abstract class for evaluating differentiable control objectives with valid vector relative degree. The dynamics `eta_dot` decompose as `eta_dot = drift(x, t) + decoupling(x, t) * u`.

The output `eta(x, t)` itself should also decompose as blocks `[eta_1(x, t), ..., eta_k(x, t)]`, corresponding to relative degree vector `[gamma_1, ..., gamma_k]`. The block `eta_i(x, t)` should contain `gamma_i` elements in increasing derivative order of the `i`-th control objective. If `eta(x, t)` is not organized in this block structure, a permutation into this structure must be provided.

### `PDOutput`

`Output` -> `PDOutput`

Abstract class for evaluating control objectives which contain proportional and derivative error components.

### `RoboticSystemOutput`

`Output` -> `AffineDynamicOutput` -> `FeedbackLinearizableOutput` -> `RoboticSystemOutput`

`Output` -> `PDOutput` -> `RoboticSystemOutput`

Abstract class for evaluating differentiable control objectives, each with relative degree 2. The dynamics `eta_dot` decompose as `eta_dot = drift(x, t) + decoupling(x, t) * u`.

Proportional and derivative error components are defined as `e_p(x, t) = y(x) - y_d(t)` and `e_d(x, t) = d/dt ( y(x) - y_d(t) )`, respectively. The output `eta(x, t)` itself should also decompose as blocks `[e_p(x, t), e_d(x, t)]`.

## Lyapunov Functions

Lyapunov functions are defined on outputs and can be evaluated given a state and time.


### `LyapunovFunction`

Abstract class for differentiable Lyapunov functions.

### `QuadraticLyapunovFunction`

`LyapunovFunction` -> `QuadraticLyapunovFunction`

Class for Lyapunov functions of the form `V(eta) = eta' * P * eta`, for positive definite `P`.

### `ControlLyapunovFunction`

`LyapunovFunction` -> `ControlLyapunovFunction`

Abstract class for differentiable Lyapunov functions for which `V_dot` can be computed as a function of state, control input, and time.

### `QuadraticControlLyapunovFunction`

`LyapunovFunction` -> `QuadraticLyapunovFunction` -> `QuadraticControlLyapunovFunction`

`LyapunovFunction` -> `ControlLyapunovFunction` -> `QuadraticControlLyapunovFunction`

Class for Lyapunov functions of the form `V(eta) = eta' * P * eta`, for positive definite `P`. `V_dot` can be decomposed as `drift(x, t) + decoupling(x, t) * u`.

### `LearnedQuadraticControlLyapunovFunction`

`LyapunovFunction` -> `QuadraticLyapunovFunction` -> `QuadraticControlLyapunovFunction` -> `LearnedQuadraticControlLyapunovFunction`

`LyapunovFunction` -> `ControlLyapunovFunction` -> `QuadraticControlLyapunovFunction` -> `LearnedQuadraticControlLyapunovFunction`

Class for Lyapunov functions of the form `V(eta) = eta' * P * eta`, for positive definite `P`. `V_dot` can be decomposed as `drift(x, t) + decoupling(x, t) * u`, where `drift` and `decoupling` are modified with additive estimation models `b(x, t)` and `a(x, t)`, respectively.

## Controllers

Controller classes specify actions as a function of state and time. The objective of a controller is specified through an Output object.

### `Controller`

Abstract class for controllers.

### `ConstantController`

`Controller` -> `ConstantController`

Class for controllers that output the same action at every state and time.

### `PDController`

`Controller` -> `PDController`

Class for controllers with actions linear in proportional and derivative terms of a PDOutput.

### `LinearizingFeedbackController`

`Controller` -> `LinearizingFeedbackController`

Class for controllers acting on FeedbackLinearizableOutput objects that invert pseudoinvert the output decoupling, subtract the output drift, and add an auxilliary control term linear in the output.

### `QPController`

`Controller` -> `QPController`

Class for controllers that compute actions by solving quadratic programs. Quadratic programs may have one constraint, which may be slacked.

### `PerturbingController`

`Controller` -> `PerturbingController`

Class for controllers that perturb nominal controllers with predetermined actions. The actions are scaled by the norm of the nominal controller, potentially offset from 0 so the perturbations may always be nonzero.

### `CombinedController`

`Controller` -> `CombinedController`

Class for controllers specified as linear combinations of other controllers.
