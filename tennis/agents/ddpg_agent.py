# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project. This agent specifically implements the DDPG algorithm.

This particularly aims to learn how to solve a multi-agent learning problem.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


from itertools import product

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from tennis.agents.agent import MainAgent
from tennis.torch_models.actor_net import ActorNetwork
from tennis.torch_models.critic_net import CriticNetwork
from tennis.replay_buffer import ReplayBuffer
from tennis import utils


class DDPGAgent(MainAgent):
    """
    This model contains my code for the DDPG agent to learn and be able to
    interact through the continuous control problem.
    """

    def __init__(self, state_size, action_size, num_instances=1, seed=13,
                 **kwargs):
        # first add additional parameters specific to DDPG

        # initialize as in base model
        super(DDPGAgent, self).__init__(state_size, action_size,
                                        num_instances, seed, **kwargs)

    def _init_alg(self):
        """
        Initialize the algorithm based on what algorithm is specified.
        """
        # initialize the actor and critics separately
        self.actor = ActorNetwork(self.state_size, self.action_size,
                                  self.inter_dims).to(self.device)
        self.actor_target = ActorNetwork(self.state_size, self.action_size,
                                         self.inter_dims).to(self.device)
        self.actor_target = utils.copy_weights(self.actor, self.actor_target)

        self.critic = CriticNetwork(self.state_size, self.action_size,
                                    self.inter_dims).to(self.device)
        self.critic_target = CriticNetwork(self.state_size, self.action_size,
                                           self.inter_dims).to(self.device)
        self.critic_target = utils.copy_weights(self.critic, self.critic_target)

        # initializer optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=self.actor_alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=self.critic_alpha,
                                           eps=1e-4)

        # initialize the replay buffer
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size,
                                   seed=self.seed)

    def get_status(self, verbose, time_diff):
        """
        Get the current state of the agent as it's training.
        """
        avg_critic_loss = np.mean(self.critic_loss_avgs)
        avg_actor_loss = np.mean(self.actor_loss_avgs)

        if verbose:
            print('----------------------------------')
            print(f'* Time taken : {time_diff} s')
            print(f'--- Critic Loss : {avg_critic_loss}')
            print(f'--- Actor Loss : {avg_actor_loss}')
            print(f'--- epsilon : {self.epsilon}')
            print('----------------------------------')

        return avg_critic_loss, avg_actor_loss

    def save_model(self, main_file):
        """
        Save the agent's underlying model(s).

        Parameters
        ----------
        file_name: str
            File name to which the agent will be saved for future use.
        """
        actor, actor_t, critic, critic_t = utils.extract_model_names(main_file)

        utils.save_model(self.actor, actor)
        utils.save_model(self.actor_target, actor_t)

        utils.save_model(self.critic, critic)
        utils.save_model(self.critic_target, critic_t)

    def load_model(self, main_file):
        """
        Load the agent's underlying model(s).

        Parameters
        ----------
        file_name: str
            File name from which the agent will be loaded.
        """
        actor, actor_t, critic, critic_t = utils.extract_model_names(main_file)

        self.actor = utils.load_model(self.actor, actor, self.device)
        self.actor_target = utils.load_model(self.actor_target, actor_t,
                                             self.device)

        self.critic = utils.load_model(self.critic, critic, self.device)
        self.critic_target = utils.load_model(self.critic_target, critic_t,
                                              self.device)

    def get_noise(self):
        """
        Sample noise to introduce randomness into the action selection process.
        """
        noise_vals = np.zeros((self.num_instances, self.action_size))
        for agent in range(self.num_instances):
            noise_vals[agent] = self.noise.sample() * self.epsilon
        #self.noise.step()
        noise_vals = torch.from_numpy(noise_vals).float().to(self.device)

        return noise_vals

    def get_action(self, states, in_train=True):
        """
        Extract the action values to be used in the environment based on the
        actor network along with a Ornstein-Uhlenbeck noise process.

        Parameters
        ----------
        states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing states information
            either in the shape (1, 33) or (batch_size, 33)

        Returns
        -------
        int
            Integer indicating the action selected by the agent based on the
            states provided.
        """
        self.actor.eval()
        with torch.no_grad():
            noise_vals = self.get_noise()
            action_vals = self.actor(states.to(self.device)) + noise_vals
            action_vals = torch.clamp(action_vals, -1, 1)
        self.actor.train()

        return action_vals

    def compute_loss(self, states, actions, next_states, rewards, dones):
        """
        Compute the loss based on the information provided and the value /
        policy parameterizations used in the algorithm.

        Parameters
        ----------
        states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing states information
        actions: np.array/torch.Tensor
            Array or Tensor singleton or batch containing actions taken
        next_states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing information about what
            state followed actions taken from the states provided by 'state'
        rewards: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        dones: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken

        Returns
        -------
        torch.float32
            Loss value (with grad) based on target and Q-value estimates.
        """
        # compute target and critic values for TD loss
        next_actor_actions = self.actor_target(next_states)
        critic_targets = self.critic_target(next_states, next_actor_actions)

        # compute loss for critic
        done_v = 1 - dones
        target_vals = rewards + self.gamma * critic_targets.squeeze(1) * done_v
        critic_vals = self.critic(states, actions)

        loss = F.mse_loss(target_vals, critic_vals)

        # then compute loss for actor
        cur_actor_actions = self.actor(states)
        policy_loss = self.critic(states, cur_actor_actions)
        policy_loss = -policy_loss.mean()

        return loss, policy_loss

    def train_critic(self, loss):
        """
        """
        self.critic.train()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

    def train_actor(self, policy_loss):
        """
        """
        self.actor.train()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def learn(self, states, actions, next_states, rewards, dones):
        """
        Learn from an experience tuple.

        Parameters
        ----------
        states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing states information
        actions: np.array/torch.Tensor
            Array or Tensor singleton or batch containing actions taken
        next_states: np.array/torch.Tensor
            Array or Tensor singleton or batch containing information about
            what state followed actions taken from the states provided by
            'state'
        rewards: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        dones: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken
        """
        self.memory.store_tuple(states, actions, next_states, rewards, dones)

        update_time_step = (self.t + 1) % self.t_update == 0
        sufficient_tuples = len(self.memory) > self.memory.batch_size

        # learn from stored tuples if enough experience and t is an update step
        if update_time_step and sufficient_tuples:

            critic_losses = []
            actor_losses = []

            for _ in range(self.num_updates):
                s, a, s_p, r, d = self.memory.sample()

                loss, policy_loss = self.compute_loss(s, a, s_p, r, d)

                # train the critic and actor separately
                self.train_critic(loss)
                self.train_actor(policy_loss)

                critic_losses.append(loss.item())
                actor_losses.append(policy_loss.item())

                self.step()

            self.critic_loss_avgs.append(np.mean(critic_losses))
            self.actor_loss_avgs.append(np.mean(actor_losses))

        # update time step counter
        self.t += 1

    def step(self):
        """
        Update state of the agent and take a step through the learning process
        to reflect experiences have been acquired and/or learned from.
        """
        # update actor target network
        self.actor_target = utils.copy_weights(self.actor, self.actor_target,
                                               self.tau)

        # update critic target network
        self.critic_target = utils.copy_weights(self.critic,
                                                self.critic_target,
                                                self.tau)
