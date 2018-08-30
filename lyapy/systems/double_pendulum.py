""" Double inverted pendulum system.

System is modeled as 2 point masses, each at the end of a massless rod. These
rods are connected. Torque input is applied at the opposite end of each rod.
State x = [theta1, theta2, theta1_dot, theta2_dot]', where
    theta1 is angle of the first mass of the double inverted pendulum in rad clockwise from upright
    theta2 is angle of the second mass of the double inverted pendulum in rad clockwise from the angle of the first mass
    theta1_dot is the angular rate in rad/sec of the first mass of the double inverted pendulum
    theta2_dot is the angular rate in rad/sec of the second mass of the double inverted pendulum
The first mass is the mass in between the 2 rods, and the second mass is the mass at the end of both the rods.

Input u = [tau_1, tau_2], where tau_1 and tau_2 are torques applied to the rods in N * m.

Equations of motion is
    J * theta_ddot = G + u
where
    J is the inertia matrix
    G represents coriolis forces and gravity
    theta_ddot is the angular acceleration [theta1_ddot, theta2_ddot] in [rad / (sec ^ 2), rad / (sec ^ 2)]

"""

from numpy import array, cos, dot, sin, sum, vstack, zeros
from .system import System

class DoublePendulum(System):
    """Double inverted pendulum system."""
    def __init__(self, m_1, m_2, g, l_1, l_2, a = None, b = None):
        """Initialize a Double Pendulum object.

        Inputs:
        First mass in kg, m_1: float
        Second mass in kg, m_2: float
        Acceleration due to gravity in m / (sec ^ 2), g: float
        Length of rod attached to both masses in m, l_1: float
        Length of rod attached to second mass in m, l_2: float
        Augmentation to actuation matrix, a: numpy array (n, ) -> numpy array (n, m)
        Augmentation to drift matrix, b: numpy array (n, ) -> numpy array (n, )
        """
        if a == None:
            def a(x):
                return zeros((4, 2))
        if b == None:
            def b(x):
                return zeros((4,))
            
        self.m_1, self.m_2, self.g, self.l_1, self.l_2 = m_1, m_2, g, l_1, l_2
        self.a, self.b = a, b

    def invertJ(self, J):
        detJ = J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]
        invJ = array([ [J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])
        return 1/detJ * invJ
    
    def calcJ(self, x):
        theta1, theta2, theta1_dot, theta2_dot = x
        J11 = (self.m_1 + self.m_2)*(self.l_1**2) + self.m_2*(self.l_2**2) + (2*self.m_2*self.l_1*self.l_2*cos(theta2))
        J12 = self.m_2*(self.l_2**2) + self.m_2*self.l_1*self.l_2*cos(theta2)
        J21 = J12
        J22 = self.m_2*(self.l_2**2)
        J = array([[J11, J12], [J21, J22]])
        return J

    def calcG(self, x):
        theta1, theta2, theta1_dot, theta2_dot = x
        G1 = self.m_2*self.l_1*self.l_2*(2*theta1_dot + theta2_dot)*theta2_dot*sin(theta2) + (self.m_1 + self.m_2)*self.g*self.l_1*sin(theta1) + self.m_2*self.g*self.l_2*sin(theta1+theta2)
        G2 = -self.m_2*self.l_1*self.l_2*(theta1_dot**2)*sin(theta2) + self.m_2*self.g*self.l_2*sin(theta1 + theta2)
        G = array([G1, G2])
        return G
    
    def drift(self, x):
        theta1, theta2, theta1_dot, theta2_dot = x
        Jinv_G = dot(self.invertJ(self.calcJ(x)), self.calcG(x).T)
        return array([theta1_dot, theta2_dot, Jinv_G[0], Jinv_G[1]]) + self.b(x)

    def act(self, x):       
        J_inv = self.invertJ(self.calcJ(x))
        return array(vstack((zeros((2, 2)), J_inv))) + self.a(x)
    


