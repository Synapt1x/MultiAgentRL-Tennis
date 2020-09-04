# -*- coding: utf-8 -*-
"""
Code implementing the functionality for a replay buffer in the DDPG algorithm in
order to solve a multi-agent (tennis) learning task.

Implementation largely re-used and slightly modified from my implementation in
my DDPG and DQN algorithms for previous projects.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tennis.datasets import TrajectoryDataset


class ReplayBuffer:
    """
    This class implements the functionality for storing experience tuples in a
    replay buffer to sample during learning steps in the DQN algorithm.
    """

    def __init__(self, buffer_size=1E6, batch_size=32, seed=13):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        np.random.seed(seed)  # seed for reproducibility

        # initalize device; use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.memory = []

    def __len__(self):
        """
        Return the size of items in the replay buffer.
        """
        return len(self.memory)

    def empty(self):
        """
        Remove all tuples stored in the replay buffer. This is useful for PPO to
        use the replay buffer as storage for trajectories.
        """
        self.memory = []

    def store_tuple(self, state, action, next_state, reward, done):
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
        # only keep the most recent tuples if memory size has been reached
        if len(self.memory) == self.buffer_size:
            num_agents = state.shape[0]
            self.memory = self.memory[num_agents:]
        full_s = state.flatten()
        full_a = action.flatten()
        full_s_p = next_state.flatten()
        self.memory.append((full_s, full_a, full_s_p, reward, done))

    def sample(self):
        """
        Extract a random sample of tuples from memory.
        """
        random_ints = np.random.choice(len(self.memory), self.batch_size,
                                       replace=False)

        raw_sample = [self.memory[i] for i in random_ints]
        exp_batch_lists = list(zip(*raw_sample))

        exp_batch = tuple(torch.from_numpy(
            np.array(exp_batch_lists[i])).float().to(self.device)
                          for i in range(len(exp_batch_lists)))

        return exp_batch

    def get_dataset(self, batch_size=32):
        """
        Extract a dataset of T timesteps over N learners stored in the buffer as
        a storage for trajectories. This is mainly used for PPO.
        """
        zipped_batch = list(zip(*self.memory))

        full_batch = list(torch.tensor(np.array(data_batch),
                          requires_grad=True).float().to(self.device)
                          for data_batch in zipped_batch)
        trajectory_dataset = TrajectoryDataset(*full_batch)

        dataset = DataLoader(trajectory_dataset, batch_size=batch_size,
                             shuffle=False)

        return dataset
