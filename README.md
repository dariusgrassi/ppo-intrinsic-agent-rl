# PPO and PPO+RND for Atari Games

This project implements and compares the performance of two reinforcement
learning agents:

1. **Proximal Policy Optimization (PPO)**: A baseline agent.
2. **PPO with Random Network Distillation (PPO+RND)**: An agent augmented with
    an intrinsic reward mechanism for improved exploration.

The motivation is to demonstrate the effectiveness of intrinsic motivation in
sparse reward environments and its potential drawbacks in dense reward settings.
To achieve this, we evaluate the performance of two
[Gymnasium](https://gymnasium.farama.org/) Atari environments: Montezuma's
Revenge v5 (sparse) and Pong (dense).

## Core Concepts

### Proximal Policy Optimization (PPO)

[PPO](https://arxiv.org/abs/1707.06347) is a policy gradient method that
optimizes a "clipped" objective function, which prevents the policy from
changing too much at each update. This leads to more stable and reliable
training compared to vanilla policy gradient methods.

### Random Network Distillation (RND)

[RND](https://arxiv.org/abs/1810.12894) is an intrinsic motivation technique
that encourages the agent to explore novel states. It consists of two neural
networks: a "target" network and a "predictor" network. The target network is
fixed and randomly initialized, while the predictor network is trained to
predict the output of the target network for a given state. The prediction error
is used as an intrinsic reward, which is high for novel states and low for
familiar states.

## Project Structure

The project is structured as follows:

- `main.py`: The main script for training the agent. It handles the training
loop, environment setup, and logging.
- `agent.py`: Defines the PPO agent, including the policy and value networks.
- `model.py`: Defines the Actor-Critic network architecture.
- `rnd.py`: Implements the Random Network Distillation (RND) module.
- `storage.py`: Implements the `RolloutStorage` class for storing trajectories.
- `env_utils.py`: Contains utility functions for environment preprocessing.

## Getting Started

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/#installation) for dependency
management.

- Python 3.12+
- `uv`

### Installing

1. Clone the repository:

    ```bash
    git clone https://github.com/dariusgrassi/ppo-intrinsic-agent-rl.git
    ```

2. Create a virtual environment and install the dependencies using `uv`:

    ```bash
    uv sync
    ```

## Usage

To train the PPO agent, run the `main.py` script:

```bash
uv run main.py
```

You can modify the hyperparameters in `main.py` to experiment with different settings.

Some key ones:

- `USE_RND`: Set to `True` to use PPO with RND, or `False` to use baseline PPO.
- `TOTAL_TIMESTEPS`: The total number of timesteps to train for.
