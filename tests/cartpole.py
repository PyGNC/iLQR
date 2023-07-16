import autograd.numpy as np
from collections import namedtuple

cart_mass = 1.0
pole_mass = 0.25
pole_length = 1.0
g = 9.8
damping_coefficient = 0.05
force_limit = 10

CartpoleConfig = namedtuple('CartpoleConfig', [
                            'cart_mass', 'pole_mass', 'pole_length', 'g', 'damping_coefficient', 'force_limit'])


def generate_cartpole_dynamics(config: CartpoleConfig):
    def cartpole_dynamics(x, u, dt):
        theta = x[1]
        z_dot = x[2]
        theta_dot = x[3]

        M = np.zeros((2, 2))
        M[1, 1] = config.cart_mass + config.pole_mass
        M[1, 2] = 0.5 * config.pole_mass * config.pole_length * np.cos(theta)
        M[2, 1] = M[1, 2]
        M[2, 2] = config.pole_mass * (config.pole_length**2) / 3

        C = np.zeros((2, 1))
        C[1] = 0.5 * config.pole_mass * config.pole_length * \
            (theta_dot**2) * np.sin(theta) - config.damping_coefficient * z_dot
        C[2] = 0.5 * config.pole_mass * config.pole_length * g * np.sin(theta)

        B = np.zeros((2, 1))
        B[1] = 1.0
        B[2] = 0.0

        Minv = np.zeros((2, 2))
        Minv[1, 1] = M[2, 2]
        Minv[1, 2] = -M[1, 2]
        Minv[2, 1] = -M[2, 1]
        Minv[2, 2] = M[1, 1]
        Minv = (1 / (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])) * Minv

        x_dot_dot = Minv * (C + B * u)

        z_dot_dot = x_dot_dot[1]
        theta_dot_dot = x_dot_dot[2]

        return [z_dot, theta_dot, z_dot_dot, theta_dot_dot]
    return cartpole_dynamics
