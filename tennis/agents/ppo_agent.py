# -*- coding: utf-8 -*-
"""
Custom RL agent for learning how to navigate through the Unity-ML environment
provided in the project. This agent specifically implements the DDPG algorithm.

This particularly aims to learn how to solve a multi-agent learning problem.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from control.agents.agent import MainAgent
from control.torch_models.actor_net import ActorNetwork
from control.torch_models.critic_net import CriticNetwork
from control.trajectory_store import TrajectoryStore
from torch.distributions import MultivariateNormal
from control import utils


class PPOAgent(MainAgent):
    """
    This model contains my code for the PPO agent to learn and be able to
    interact through the continuous control problem.
    """

    def __init__(self, state_size, action_size, num_instances=1, seed=13,
                 **kwargs):
        # first add additional parameters specific to PPO
        self.eps_clip = kwargs.get('eps_clip', 0.05)
        self.num_updates = kwargs.get('num_updates', 10)
        self.t_update = kwargs.get('t_update', 5)
        self.variance_decay = kwargs.get('variance_decay', 0.99)
        self.noise = None
        self.t = 0

        # initialize as in base model
        super(PPOAgent, self).__init__(state_size, action_size,
                                       num_instances, seed, **kwargs)

        self.action_variance = kwargs.get('action_variance', 0.25)
        self.action_variances = self.set_action_variances(multiplier=1.0)

    def _init_alg(self):
        """
        Initialize the algorithm based on what algorithm is specified.
        """
        self.policy = ActorNetwork(self.state_size, self.action_size,
                                   self.inter_dims).to(self.device)
        self.prev_policy = ActorNetwork(self.state_size, self.action_size,
                                        self.inter_dims).to(self.device)
        self.critic = CriticNetwork(
            self.state_size, 0, self.inter_dims).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.actor_alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=self.critic_alpha)

        # trajectory storage
        self.trajectories = TrajectoryStore(self.num_instances, self.t_update,
                                            self.state_size, self.action_size)

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
            print(f'--- action var : {self.action_variance}')
            print('----------------------------------')

        return avg_critic_loss, avg_actor_loss

    def set_action_variances(self, multiplier=1.0):
        """
        Set the value for the action variance to use when sampling actions from
        the stochastic policy maintained by PPO.
        """
        self.action_variance = self.action_variance * multiplier
        variance_array = np.array([self.action_variance] * self.action_size)

        return torch.diag(
            torch.from_numpy(variance_array)).float().to(self.device)

    def compute_discounted_rewards(self, rewards):
        """
        Compute discounted rewards for a set of sequential rewards.
        """
        discounts = torch.tensor(
            np.array([self.gamma ** i for i in range(self.t_update)]),
            requires_grad=True).float().to(self.device)

        indices = torch.LongTensor(np.linspace(self.t_update - 1, 0,
                                               self.t_update)).to(self.device)
        reversed_rewards = rewards.index_select(0, indices)
        final_rewards = torch.cumsum(reversed_rewards, 0)
        discounted_rewards = (discounts * final_rewards).float().to(self.device)

        forward_discounts = discounted_rewards.index_select(0, indices)

        return forward_discounts

    def get_action_distribution(self, states, in_train=True):
        """
        Extract the probability distribution for actions.
        """
        policy = self.policy if in_train else self.prev_policy
        if not in_train:
            policy.eval()
            with torch.no_grad():
                action_probs = policy(states.to(self.device))
                distribution = MultivariateNormal(
                    action_probs, self.action_variances)

            policy.train()
        else:
            action_probs = policy(states.to(self.device))
            distribution = MultivariateNormal(action_probs,
                                              self.action_variances)

        return distribution

    def get_noise(self):
        """
        Sample noise to introduce randomness into the action selection process.
        """
        noise_vals = np.array(self.noise.sample())
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
        distribution = self.get_action_distribution(states, in_train)
        actions = distribution.sample().to(self.device)

        noise_vals = torch.stack(self.get_noise())
        action = actions + noise_vals
        actions = torch.clamp(actions, -1, 1)

        return actions

    def clipped_surrogate(self, advantages, update_logprobs, logprobs):
        """
        Compute the clipped surrogate function with the provided trajectories.
        """
        ratio = torch.exp(update_logprobs - logprobs)
        clamped_ratio = torch.clamp(ratio, 1 - self.eps_clip,
                                    1 + self.eps_clip).to(self.device)

        # compute the surrogate function values
        l_surr = torch.mean(ratio * advantages)
        clamp_l_surr = torch.mean(clamped_ratio * advantages)
        final_surr = -torch.min(l_surr, clamp_l_surr).requires_grad_()

        return final_surr

    def compute_advantages(self, states, actions, next_states, rewards, dones):
        """
        Compute the advantages based on the rewards and state values as provided
        by the critic.

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
            Advantage values based on rewards and state value.
        """
        discounted_rewards = self.compute_discounted_rewards(rewards)
        state_vals = self.critic(states).float().squeeze(1)

        advantages = discounted_rewards - state_vals

        # compute the state value loss
        critic_loss = F.mse_loss(state_vals, rewards)

        return advantages, critic_loss

    def update_policy(self, loss):
        """
        Update the policy using the provided loss computed from the policy
        gradient via the surrogate function bound.

        Parameters
        ----------
        loss : torch.loss
            Loss function encapsulating the loss value to update the policy.
        """
        self.policy.train()
        self.policy_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.policy_optimizer.step()

    def update_value(self, critic_loss):
        """
        Update the value function critic providing the baseline for advantages.

        Parameters
        ----------
        loss : torch.loss
            Loss function encapsulating the loss value to update the policy.
        """
        self.critic.train()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def get_logs(self, s, a):
        """
        Compute the action probability logs using both policies.
        """
        prev_prob_dist = self.get_action_distribution(s, in_train=False)
        prob_dist = self.get_action_distribution(s, in_train=False)

        prev_logs = prev_prob_dist.log_prob(a)
        logs = prob_dist.log_prob(a)

        return prev_logs, logs

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
            Array or Tensor singleton or batch containing information about what
            state followed actions taken from the states provided by 'state'
        rewards: np.array/torch.Tensor
            Array or Tensor singleton or batch containing reward information
        dones: np.array/torch.Tensor
            Array or Tensor singleton or batch representing whether or not the
            episode ended after actions were taken
        """
        self.trajectories.store_tuple(states, actions, next_states,
                                      rewards, dones, self.t)

        if self.t == self.t_update - 1:
            update_dataset = self.trajectories.get_dataset()

            critic_losses = []
            actor_losses = []

            for epoch in range(self.num_updates):
                for batch_i, exp_tuples in enumerate(update_dataset):
                    all_s, all_a, all_s_n, all_r, all_d = exp_tuples
                    for agent in range(self.num_instances):
                        s = all_s[:, agent, :]
                        a = all_a[:, agent, :]
                        s_n = all_s_n[:, agent, :]
                        r = all_r[:, agent]
                        d = all_d[:, agent]

                        advantages, critic_loss = self.compute_advantages(
                            s, a, s_n, r, d)

                        prev_logs, action_logs = self.get_logs(s, a)
                        loss = self.clipped_surrogate(advantages, action_logs,
                                                      prev_logs)

                        self.update_policy(loss)
                        self.update_value(critic_loss)

                        critic_losses.append(critic_loss.item())
                        actor_losses.append(loss.item())

            self.critic_loss_avgs.append(np.mean(critic_losses))
            self.actor_loss_avgs.append(np.mean(actor_losses))

            self.step()
            self.trajectories.empty()

        self.t += 1

    def step(self):
        """
        Update state of the agent and take a step through the learning process
        to reflect experiences have been acquired and/or learned from.
        """
        # update actor target network
        self.prev_policy = utils.copy_weights(self.policy, self.prev_policy)

        # decay epsilon for random noise
        self.epsilon = np.max([self.epsilon * self.epsilon_decay,
                               self.epsilon_min])

        # update action variance to choose more selectively
        self.action_variances = self.set_action_variances(self.variance_decay)

        self.t = -1