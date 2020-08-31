# Deep RL Continuous Control - Reacher in Unity-ML


Project code for solving a multi-agent learning problem, specifically solving the Tennis environment by training multiple agents to play a two-player game of tennis in Unity-ML.
Part of my work for the Deep RL Nanodegree on Udacity.

## Author

Chris Cadonic

chriscadonic@gmail.com

## Background

**
A full report describing much of my approach, results, and potential improvements can be found in the file [docs/Report.pdf](docs/Report.pdf).**

More information to be placed here when more is understood about the
environment.

## Setup and Running

### Setup

Just as in outlined in [the DRLND repository](https://github.com/udacity/deep-reinforcement-learning#dependencies), the following steps can be used to setup the environment:

1. Setup an `anaconda` environment (optional):
```
conda create --name drlnd python=3.6
```
and then activate this environment using:
```
conda activate drlnd
```
in MacOS and Linux or
```
activate drlnd
```
in Windows.

2. As in the DRLND repository, install dependencies using:
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

3. Setup the Unity environment

This environment is specifically provided by Udacity to contain the learning environment. A link to a file will be added here once verified that
such a link can be provided openly.

With the Unity environment acquired, the directory structure for this project should then be:

```
configs/...
docs/...
envs/...  <-- place the unzipped Unity environment directories here
models/...
tennis/...  <-- main code for project
output/...
README.md
runner.py
requirements.txt
```
Main code in the `tennis` directory is organized as follows:
```
tennis/
    agents/
        agent.py             <-- base code for running an agent (defaults behaviour is a random agent)
        d4pg_agent.py        <-- code for running the D4PG algorithm
        ddpg_agent.py        <-- code for running the DDPG algorithm
        ppo_agent.py         <-- code for running the PPO algorithm
    tennis_main.py           <-- code for interfacing the agents and the environment
                                  and running episodes in the environment
    torch_models/
        simple_linear.py     <-- torch model for DDPG using a set of linearly connected layers
```

### Running

```
python runner.py
```
from the root of this project repository. Arguments can be provided as follows:
- `-t` (or `--train`): train the model with the provided configuration file (Defaults to False),
- `-c` (or `--config`): specify the configuration to use for training the model (Defaults to `configs/default_config.yaml`.

Thus, running the model to show inference using the final trained model without visualization can be run using:
```
python runner.py
```
or with visualization using:
```
python runner.py -c configs/default_vis_config.yaml
```

The model can also be retrained if one wishes by passing the `-t` or `--train` flag. Be careful as this will overwrite any output in the `output/` directory and saved models in the `models` directory, as specified by the configuration file.

If one wishes to change parameters, then you can create a new configuration file, or modify an existing configuration file, and provide parameters in the following format:
```
# general parameters
```

**To be completed**

## Results

**To be completed**
