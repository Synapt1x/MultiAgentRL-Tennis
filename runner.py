# -*- coding: utf-8 -*-
"""
This code is the main runner CLI for the relevant project code. This will
likely be static for the majority of my projects in order to afford a simple and
reliable CLI for accessing my project code.

As such, this code is slightly  modified from my implementation of DDPG in solving a continuous control problem (Reacher) from project 2.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import yaml
import argparse

from tennis.tennis_main import TennisMain


# global constants
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(CUR_DIR, 'configs', 'default_config.yaml')


def parse_args():
    """
    Parse provided arguments from the command line.
    """
    arg_parser = argparse.ArgumentParser(
        description="Argument parsing for accessing andrunning my deep "\
            "reinforcement learning projects")

    # command-line arguments
    arg_parser.add_argument('-t', '--train', action='store_true',
                            default=False,
                            help='Flag to indicate whether to train or not')
    arg_parser.add_argument('-c', '--config', dest='config_file',
                            type=str, default=DEFAULT_CONFIG)
    arg_parser.add_argument('-v', '--verbose', action='store_true',
                            default=False,
                            help='Verbose flag for printing out progress')
    args = vars(arg_parser.parse_args())

    return args


def load_config(path):
    """
    Load the configuration file that will specify the properties and parameters
    that may change in the general problem environment and/or the underlying RL
    agent/algorithm.
    """
    with open(path, 'r') as config:
        config_data = yaml.safe_load(config)

    return config_data


def parse_config(config_data):
    """
    Parse the configuration arguments.
    """
    model_file = config_data.pop('model_file')
    model_params = config_data.pop('model_params')

    return config_data, model_file, model_params


def main(model_file, model_params, train, config_data, verbose):
    """
    Main runner for the code CLI.
    """
    tennis_prob = TennisMain(model_params=model_params, **config_data,
                             verbose=verbose)

    if train:
        tennis_prob.train_agent()
        tennis_prob.save_model(model_file)
    else:
        tennis_prob.load_model(model_file)
        tennis_prob.train_agent(train_mode=False)


if __name__ == '__main__':
    # load params
    args = parse_args()

    train_on = args.get('train', False)
    config_file = args.get('config_file', DEFAULT_CONFIG)
    verbose = args.get('verbose', False)
    config_args = load_config(config_file)
    config_data, model_file, model_params = parse_config(config_args)

    # run the model with the provided parameters
    main(model_file, model_params, train_on, config_data, verbose)
