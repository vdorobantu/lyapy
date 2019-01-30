"""All system classes.

AffineControlSystem - Base class for affine control systems of the form x_dot = f(x) + g(x) * u.
ControlSystem - Base class for control systems of the form x_dot = f(x, u, t).
System - Base class for dynamical systems of the form x_dot = f(t, x).
"""

from .affine_control_system import AffineControlSystem
from .control_system import ControlSystem
from .system import System
