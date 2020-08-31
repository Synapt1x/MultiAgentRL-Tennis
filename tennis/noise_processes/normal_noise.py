# -*- coding: utf-8 -*-
"""
This code contains a wrapper for simply serving random normal samples for
extracting noise from a Gaussian distribution. Noise itself follows an epsilon
decay to reduce the amount of random noise injected into actions.

Code modified from my implementation of DDPG in solving a continuous control problem (Reacher) from project 2.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np


class NormalNoise():
    """
    Noise wraps sampling from a standard Gaussian (Normal) distribution.
    """

    def __init__(self, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
                 noise_variance=0.3, action_size=4, seed=13):
        self.original_epsilon = epsilon

        self.action_size = action_size

        # set decay parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # set variance for the N(0, sd)
        self.noise_variance = noise_variance

        # seed only on construction not every reset
        self.seed = seed
        np.random.seed(self.seed)

        self.init_process()

    def init_process(self):
        """
        Initialize or reset the Ornstein-Uhlenbeck process.
        """
        self.epsilon = self.original_epsilon

    def sample(self):
        """
        Sample a random number and multiply by epsilon to scale noise.
        """
        noise_val = np.random.randn(self.action_size) * self.noise_variance
        scaled_val = noise_val * self.epsilon

        return scaled_val

    def step(self):
        """
        Update epsilon.
        """
        update_val = self.epsilon * self.epsilon_decay
        self.epsilon = np.max([update_val, self.epsilon_min])


if __name__ == '__main__':

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt


    print('Testing Normal process noise and saving to '
          'test_normal.png')

    t_end = 1
    t_start = 0
    n = 1000
    t = np.linspace(t_start, t_end, n)
    y = []

    noise = NormalNoise(epsilon_decay=0.995, noise_variance=0.5)
    for i in range(n):
        y.append(noise.sample())
        noise.step()

    plt.plot(t, y)
    plt.savefig('test_normal.png')