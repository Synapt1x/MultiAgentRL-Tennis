# -*- coding: utf-8 -*-
"""
Main code for the multi-agent (tennis) learning task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse
import datetime
import pickle
import time

from unityagents import UnityEnvironment
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use backend for saving plots only
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from tennis import utils


class TennisMain:
    """
    This code contains functionality for running the Tennis environment and
    running the code as per training, showing performance, and loading/saving my
    models.
    """

    def __init__(self, file_path, alg, graph_file, model_params,
                 frame_time=0.075, max_episodes=1E5, max_iterations=1E5,
                 t_update=20, num_updates=10, verbose=False):
        self.frame_time = frame_time
        self.max_iterations = max_iterations
        self.max_episodes = max_episodes
        self.graph_file = graph_file
        self.verbose = verbose

        self.env, self.brain_name, self.brain = self._init_env(file_path)
        self.agent = self._init_agent(alg, model_params, t_update, num_updates)

        self.score_store = []
        self.critic_loss_store = []
        self.actor_loss_store = []
        self.average_scores = []
        self.solved_score = 30

    def _init_env(self, file_path):
        """
        Initialize the Unity-ML Tennis environment.
        """
        env = UnityEnvironment(file_name=file_path)
        brain_name = env.brain_names[0]
        first_brain = env.brains[brain_name]

        return env, brain_name, first_brain

    def _init_agent(self, alg, model_params, t_update, num_updates):
        """
        Initialize the custom model utilized by the agent.
        """
        # extract state and action information
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        num_agents = len(env_info.agents)
        state_size = len(env_info.vector_observations[0])
        action_size = self.brain.vector_action_space_size

        if alg.lower() == 'maddpg':
            # init DDPG
            from tennis.agents.maddpg_agent import MADDPGAgent
            return MADDPGAgent(**model_params, state_size=state_size,
                               action_size=action_size, num_instances=num_agents,
                               t_update=t_update, num_updates=num_updates)
        else:
            # default to random
            from tennis.agents.agent import MainAgent
            return MainAgent(**model_params, state_size=state_size,
                             action_size=action_size, num_instances=num_agents,
                             t_update=t_update, num_updates=num_updates)

    def _update_scores(self, scores):
        """
        Store scores from an episode into running storage.
        """
        # store average over agents
        self.score_store.append(np.max(scores))

        # also store average over last 100 episodes over agent average
        score_avg = np.mean(self.score_store[-100:])
        self.average_scores.append(score_avg)

    def _store_losses(self, time_diff):
        """
        Print out and store scores from losses.
        """
        avg_critic_loss, avg_actor_loss = self.agent.get_status(self.verbose,
                                                                time_diff)

        self.critic_loss_store.append(avg_critic_loss)
        self.actor_loss_store.append(avg_actor_loss)

    def save_model(self, file_name):
        """
        Save the model to the file name specified.

        Parameters
        ----------
        file_name: str
            File name to which the agent will be saved for future use.
        """
        model_dir = os.path.dirname(file_name)
        os.makedirs(model_dir, exist_ok=True)
        self.agent.save_model(file_name)

    def load_model(self, file_name):
        """
        Load the model specified.

        Parameters
        ----------
        file_name: str
            File name from which the agent will be loaded.
        """
        self.agent.load_model(file_name)

    def save_training_plot(self, first_solved):
        """
        Plot training performance through episodes.

        Parameters
        ----------
        first_solved: int
            Episode number at which the agent solved the continuous control
            problem by achieving an average score of +30.
        """
        num_eval = len(self.average_scores)

        if num_eval > 100:
            # Set up plot file and directory names
            out_dir, cur_date = utils.get_output_dir()
            plot_file = os.path.join(
                out_dir, self.graph_file.replace(
                    '.png', f'-date-{cur_date}.png'))

            # plot and save the plot file
            fig = plt.figure(figsize=(12, 8))

            plt.plot(self.score_store, linewidth=1, alpha=0.4,
                     label='raw_episode_score')
            plt.plot(self.average_scores, linewidth=2,
                     label='100_episode_avg_score')
            plt.title(f'Average Score Over Recent 100 Episodes During Training',
                      fontsize=20)

            plt.xlabel('Episode', fontsize=16)
            plt.ylabel('Average Score', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.xlim([0, num_eval])
            plt.ylim([0, np.max(self.score_store)])

            # plot indicator for solved iteration
            if first_solved > 0:
                min_val = np.min(self.average_scores)
                plt.axhline(y=13, color='g', linewidth=1, linestyle='--')
                ax = fig.gca()

                ax.add_artist(Ellipse((first_solved, self.solved_score),
                                      width=5, height=1, facecolor='None',
                                      edgecolor='r', linewidth=3, zorder=10))
                plt.text(first_solved + 10, int(self.solved_score * 0.8),
                         f'Solved in {first_solved} episodes', color='r',
                         fontsize=14)

            plt.legend(fontsize=12)

            plt.savefig(plot_file)

            print(f'Training plot saved to {plot_file}')
        else:
            print('Not enough average scores computed. Skipping plotting.')

    def save_loss_plots(self):
        """
        Plot and save figures for training loss for both critic and actor
        losses.

        Parameters
        ----------
        first_solved: int
            Episode number at which the agent solved the continuous control
            problem by achieving an average score of +30.
        """
        num_eval = len(self.critic_loss_store)

        if num_eval > 100:
            # Set up plot file and directory names
            out_dir, cur_date = utils.get_output_dir()
            critic_plot_file = os.path.join(
                out_dir, self.graph_file.replace(
                    '.png', f'-critic-loss-date-{cur_date}.png'))
            actor_plot_file = os.path.join(
                out_dir, self.graph_file.replace(
                    '.png', f'-actor-loss-date-{cur_date}.png'))

            utils.plot_loss(self.critic_loss_store, critic_plot_file,
                            label='critic loss',
                            title='Average Episode Critic Loss During Training')
            utils.plot_loss(self.actor_loss_store, actor_plot_file,
                            label='actor loss',
                            title='Average Episode Actor Loss During Training')
        else:
            print('Not enough loss values computed. Skipping plotting.')

    def save_results(self):
        """
        Save training averages over time.
        """
        num_eval = len(self.average_scores)

        if num_eval > 100:
            # Save results
            out_dir, cur_date = utils.get_output_dir()
            res_file = os.path.join(out_dir,
                                    f'results-file-{cur_date}.pkl')

            with open(res_file, 'wb') as o_file:
                pickle.dump(self.average_scores, o_file)

            print(f'Training results saved to {res_file}')
        else:
            print('Not enough average score computed. Skipping saving results.')

    def run_episode(self, train_mode=True):
        """
        Run an episode of interaction in the Unity-ML Tennis environment.

        Parameters
        ----------
        train_mode: bool
            Flag to indicate whether or not the agent will be training or just
            running inference.
        """
        iteration = 0

        # initiate interaction and learning in environment
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        states = env_info.vector_observations

        # get the number of agents and initialize a score for each
        scores = np.zeros(self.agent.num_instances)

        # reset noise process for Ornstein-Uhlenbeck process
        self.agent.noise.init_process()

        while iteration < self.max_iterations:
            # first have the agent act and evaluate state
            np_actions = np.zeros((self.agent.num_instances,
                                   self.agent.action_size))
            raw_actions = []
            for i, state_i in enumerate(states):
                actions_i = self.agent.get_action(utils.to_tensor(state_i),
                                                  agent_num=i,
                                                  in_train=False)
                raw_actions.append(actions_i)
                np_actions[i] = actions_i.cpu().numpy()
            actions = utils.to_tensor(raw_actions)
            env_info = self.env.step(np_actions)[self.brain_name]
            next_states, rewards, dones = utils.eval_state(env_info)

            # learn from experience tuple batch
            if train_mode:
                self.agent.learn(states, np_actions, next_states, rewards,
                                 dones)

            # increment score and compute average
            scores += rewards
            states = next_states

            if np.any(np.array(dones)):
                break
            time.sleep(self.frame_time)

            # print average score as training progresses
            iteration += 1

        # decay epsilon
        self.agent.decay_epsilon()

        return scores

    def train_agent(self, train_mode=True):
        """
        Train an agent by running learning episodes in the Tennis task.

        Parameters
        ----------
        train_mode: bool
            Flag to indicate whether or not the agent will be training or just
            running inference.
        """
        episode = 1
        try:
            # run episodes
            if not train_mode:
                self.max_episodes = np.max([100, self.max_episodes])
                self.agent.epsilon = self.epsilon_min
            while episode < self.max_episodes:
                start_t = time.time()
                scores = self.run_episode(train_mode=train_mode)
                end_t = time.time()

                self._update_scores(scores)

                print(f'* Episode {episode} completed * max: {np.max(scores)} *')
                self._store_losses(end_t - start_t)

                episode += 1

        except KeyboardInterrupt:
            print("Exiting learning gracefully...")
        finally:
            if train_mode:
                first_solved = np.argmax(
                    np.array(self.average_scores) >= self.solved_score)
                if first_solved > 0:
                    print(f'*** SOLVED IN {first_solved} EPISODES ***')
                self.save_training_plot(first_solved)
                self.save_loss_plots()
                self.save_results()

            self.env.close()
