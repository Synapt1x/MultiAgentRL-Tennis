# -*- coding: utf-8 -*-
"""
Simple torch model that is a set of fully connected layers.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
from torch import nn

from tennis.utils import compute_bound


class ActorNetwork(nn.Module):
    """
    Torch model containing a set of dense fully connected layers.
    """

    def __init__(self, state_size, action_size, inter_dims=None,
                 use_batch_norm=False, seed=13):
        if inter_dims is None:
            self.inter_dims = [64, 256]
        else:
            self.inter_dims = inter_dims

        self.state_size = state_size
        self.action_size = action_size
        self.use_batch_norm = use_batch_norm

        # set the seed
        self.seed = seed
        torch.manual_seed(self.seed)

        super(ActorNetwork, self).__init__()

        # initialize the architecture
        self._init_model()
        #self._init_weights()

    def _init_model(self):
        """
        Define the architecture and all layers in the model.
        """
        self.input = nn.Linear(self.state_size, self.inter_dims[0])

        if self.use_batch_norm:
            self.input_batch = nn.BatchNorm1d(self.inter_dims[0])

        hidden_layers = []
        batch_norms = []

        for dim_i, hidden_dim in enumerate(self.inter_dims[1:]):
            prev_dim = self.inter_dims[dim_i]
            hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.hidden_layers = nn.ModuleList(hidden_layers)
        if self.use_batch_norm:
            self.hidden_batch_norms = nn.ModuleList(batch_norms)
        self.output = nn.Linear(self.inter_dims[-1], self.action_size)

    def _init_weights(self):
        for layer_num, layer in enumerate(self.hidden_layers):
            layer_size = self.inter_dims[layer_num]
            b = compute_bound(layer_size)
            layer.weight.data.uniform_(-b, b)
        self.output.weight.data.uniform_(-3e-5, 3e-5)

    def forward(self, state):
        """
        Define the forward-pass for data through the model.

        Parameters
        ----------
        state: torch.Tensor
            A 37-length Torch.Tensor containing a state vector to be run through
            the network.

        Returns
        -------
        torch.Tensor
            Tensor containing output action values determined by the network.
        """
        if self.use_batch_norm:
            data_x = self.input_batch(torch.relu(self.input(state.float())))

            for layer, batch_norm in zip(self.hidden_layers,
                                         self.hidden_batch_norms):
                data_x = batch_norm(torch.relu(layer(data_x)))
                data_x = torch.relu(layer(data_x))
        else:
            data_x = torch.relu(self.input(state.float()))

            for layer in self.hidden_layers:
                data_x = torch.relu(layer(data_x))

        action_values = torch.tanh(self.output(data_x))

        return action_values
