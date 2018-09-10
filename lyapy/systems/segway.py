"""Planar Segway system.

System is modeled as a pair of linked wheels witb a rigid frame attached to a
point mass payload. The wheels roll without slipping, the system is modeled
without friction. State x = [x, x_dot, theta, theta_dot]', where x is the
position of the Segway base in m, x_dot is the velocity in m / sec, theta is the
angle of the frame in rad clockwise from upright, and theta_dot is the angular
rate in rad / sec. Input u = [tau]', where tau is motor torque in N * m.
"""

from numpy import array, cos, sin

from .system import System

class Segway(System):
    def __init__(self, m_b, m_w, J_w, c_2, B_2, R, K, r, g, h, m_p):
        """Initialize a Segway object

        Inputs:
        Mass of the frame, m_b: float
        Mass of the wheel, m_w: float
        Inertia of the wheel, J_w: float
        Height of the frame CG, c_2: float
        Inertia of the frame, B_2: float
        Torque constant of the motors, K: float
        Electrical resistance of the motors, R: float
        Radius of the wheels, r: float
        Gravity constant, g: float
        Height of the payload, h: float
        Mass of the payload, m_p: float
        """

        m_t = m_p + m_b
        c_f = (m_p * h + m_b * c_2) / m_t
        J_f1 = B_2 + m_p * (h ** 2) + m_b * (c_2 ** 2)

        self.N_1 = (2 * J_f1 * (K ** 2)) / (R * r)
        self.N_2 = -J_f1 * c_f * m_t * r
        self.N_3 = 2 * c_f * m_t * (K ** 2) / R
        self.N_4 = ((c_f * m_t) ** 2) * r * g
        self.N_5 = -J_f1 * (2 * J_w / r + (m_t + 2 * m_w) * r)
        self.N_6 = ((c_f * m_t) ** 2) * r

        self.M_1 = 2 * (K ** 2) * (2 * J_w + (m_t + 2 * m_w) * (r ** 2))
        self.M_2 = 2 * (K ** 2) * c_f * m_t * r
        self.M_3 = r * g * c_f * m_t * (2 * J_w + (m_t + 2 * m_w) * (r ** 2)) * R
        self.M_4 = -((c_f * m_t) ** 2) * (r ** 3) * R
        self.M_5 = r * R * J_f1 * (2 * J_w + (m_t + 2 * m_w) * (r ** 2))
        self.M_6 = -(r ** 3) * R * ((c_f * m_t) ** 2)

        self.T_1 = -2 * J_f1 * K / R
        self.T_2 = -2 * c_f * m_t * K * r / R
        self.T_3 = self.N_5
        self.T_4 = self.N_6

        self.S_1 = -2 * K * r * (2 * J_w + (m_t + 2 * m_w) * (r ** 2))
        self.S_2 = -2 * (r ** 2) * c_f * m_t * K
        self.S_3 = self.M_5
        self.S_4 = self.M_6


    def drift(self, x):
        _, theta, x_dot, theta_dot = x
        f_1 = x_dot
        f_2 = theta_dot
        f_3 = (self.N_1 * x_dot + self.N_2 * (theta_dot ** 2) * sin(theta) + self.N_3 * x_dot * cos(theta) + self.N_4 * cos(theta) * sin(theta)) / (self.N_5 + self.N_6 * (cos(theta) ** 2))
        f_4 = (self.M_1 * x_dot + self.M_2 * x_dot * cos(theta) + self.M_3 * sin(theta) + self.M_4 * (theta_dot ** 2) * sin(theta) * cos(theta)) / (self.M_5 + self.M_6 * (cos(theta) ** 2))
        return array([f_1, f_2, f_3, f_4])

    def act(self, x):
        _, theta, _, _ = x
        g_1 = 0
        g_2 = 0
        g_3 = (self.T_1 + self.T_2 * cos(theta)) / (self.T_3 + self.T_4 * (cos(theta) ** 2))
        g_4 = (self.S_1 + self.S_2 * cos(theta)) / (self.S_3 + self.S_4 * (cos(theta) ** 2))
        return array([[g_1], [g_2], [g_3], [g_4]])
