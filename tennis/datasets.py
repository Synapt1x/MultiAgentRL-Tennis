# -*- coding: utf-8 -*-
"""
This is a custom dataset is for holding batches of trajectories. Specific use
is for PPO.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Custom dataset for loading batch data training the PPO algorithm.
    """

    def __init__(self, states, actions, next_states, rewards, dones):
        # initalize device; use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.states = torch.from_numpy(states).float().to(self.device)
        self.actions = torch.from_numpy(actions).float().to(self.device)
        self.next_states = torch.from_numpy(next_states).float().to(self.device)
        self.rewards = torch.from_numpy(rewards).float().to(self.device)
        self.dones = torch.from_numpy(dones).float().to(self.device)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return tuple([self.states[idx],
                      self.actions[idx],
                      self.next_states[idx],
                      self.rewards[idx],
                      self.dones[idx]])