# -*- coding: utf-8 -*-
"""
Code implementing the functionality for storing trajectories in the PPO or any
other algorithm that leverages n-step bootstrapping.

Code slightly modified from my implementation of DDPG in solving a continuous control problem (Reacher) from project 2.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tennis.datasets import TrajectoryDataset


class TrajectoryStore:
    """
    This class implements storage of trajectories.
    """

    def __init__(self, num_agents, t_update, state_size, action_size):
        self.num_agents = num_agents
        self.t_update = t_update
        self.state_size = state_size
        self.action_size = action_size

        self.empty()

    def __len__(self):
        """
        Return the size of items in the trajectory storage.
        """
        return self.num_stored

    def empty(self):
        """
        Remove all tuples stored in the replay buffer. This is useful for PPO to
        use the replay buffer as storage for trajectories.
        """
        self.states = np.zeros((self.t_update, self.num_agents,
                               self.state_size))
        self.actions = np.zeros((self.t_update, self.num_agents,
                                self.action_size))
        self.next_states = np.zeros((self.t_update, self.num_agents,
                                    self.state_size))
        self.rewards = np.zeros((self.t_update, self.num_agents))
        self.dones = np.zeros((self.t_update, self.num_agents))

        self.num_stored = 0

    def store_tuple(self, state, action, next_state, reward, done, t):
        """
        Add the experience tuple to memory.

        Parameters
        ----------
        state: np.array/torch.Tensor
            Tensor singleton containing state information
        action: np.array/torch.Tensor
            Tensor singleton containing the action taken from state
        next_state: np.array/torch.Tensor
            Tensor singleton containing information about what state followed
            the action taken from the state provided by 'state'
        reward: np.array/torch.Tensor
            Tensor singleton containing reward information
        done: np.array/torch.Tensor
            Tensor singleton representing whether or not the episode ended after
            action was taken
        """
        # store values into the pre-initialized zero arrays
        self.states[t, :, :] = state
        self.actions[t, :, :] = action
        self.next_states[t, :, :] = next_state
        self.rewards[t, :] = reward
        self.dones[t, :] = done

        self.num_stored += 1

    def get_dataset(self):
        """
        Extract a dataset of T timesteps over N learners stored in the buffer as
        a storage for trajectories. This is mainly used for PPO.
        """
        trajectory_dataset = TrajectoryDataset(self.states,
                                               self.actions,
                                               self.next_states,
                                               self.rewards,
                                               self.dones)

        dataset = DataLoader(trajectory_dataset, batch_size=self.t_update,
                             shuffle=False)

        return dataset