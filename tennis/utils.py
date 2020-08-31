# -*- coding: utf-8 -*-
"""
Helper functionality for the multi-agent (tennis) learning task.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt


def eval_state(curr_env_info):
    """
    Evaluate a provided game state.
    """
    s = curr_env_info.vector_observations
    r = curr_env_info.rewards
    d = np.array(curr_env_info.local_done).astype(int)  # convert bool->int

    return s, r, d


def print_progress(iteration, score_avg):
    """
    Helper method for printing out the state of the game after completion.
    """
    print(f"Average score so far: {score_avg}")


def print_on_close(score):
    """
    Helper method for printing out the state of the game after completion.
    """
    print(f"Final Score: {score}")


def get_output_dir():
    """
    Return the output file path.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, os.pardir, 'output')
    cur_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    return out_dir, cur_date


def to_tensor(in_arr, device=torch.device('cpu'), dtype='float'):
    """
    Convert the provided array to a torch tensor.
    """
    tensor = torch.from_numpy(in_arr)

    if dtype == 'float':
        tensor = tensor.float()

    return tensor.to(device)


def compute_bound(layer_fanin):
    """
    Compute the uniform initialization bound value using the fan-in (input size)
    for a layer. This is used for initialization of all layers except output
    layers as discussed in Lillicrap et al.
    """
    bound = 1 / np.sqrt(layer_fanin)

    return bound


def copy_weights(network, target_network, tau=1.0):
    """
    Copy weights from network to target_network. If tau is provided then compute
    a soft update, else assume a complete copy of weights between networks.
    """
    for t_param, p_param in zip(target_network.parameters(),
                                network.parameters()):
        update_p = tau * p_param.data
        target_p = (1.0 - tau) * t_param.data
        t_param.data.copy_(update_p + target_p)

    return target_network


def extract_model_names(main_file):
    """
    Extract the final names of files to which model weights will be saved.

    Parameters
    ----------
    main_file:
        The base file name for the output model files.
    """
    file_split = main_file.split('.')[:-1]
    actor_name = ''.join(file_split + ['-actor.pkl'])
    actor_t_name = ''.join(file_split + ['-actor-target.pkl'])

    critic_name = ''.join(file_split + ['-critic.pkl'])
    critic_t_name = ''.join(file_split + ['-critic-target.pkl'])

    return actor_name, actor_t_name, critic_name, critic_t_name


def save_model(model, file_name):
    """
    Save the provided model to the provided file_name.

    Parameters
    ----------
    file_name: str
        File name to which the agent will be saved for future use.
    """
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name, device):
    """
    Load the parameters for the specified model.

    Parameters
    ----------
    file_name: str
        File name from which the agent will be loaded.
    """
    if device.type == 'cpu':
        model.load_state_dict(torch.load(file_name,
                                         map_location=device.type))
    else:
        model.load_state_dict(torch.load(file_name))
    model.eval()

    return model


def plot_loss(loss_data, plot_file, label, title):
    """
    Plot a curve of the loss over training.
    """
    # plot and save the plot file
    fig = plt.figure(figsize=(12, 8))

    plt.plot(loss_data, linewidth=2, label=label)
    plt.title(title, fontsize=20)

    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Average Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim([0, len(loss_data)])
    plt.ylim([0, np.max(loss_data)])

    plt.savefig(plot_file)

    print(f'Loss plot saved to {plot_file}')

    plt.close()