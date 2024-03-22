# RL Taxi Driver

This project implements reinforcement learning algorithms to train an agent to navigate and pick up passengers in the Taxi-v3 environment provided by OpenAI Gym.

## Overview

The project consists of several components:

- `rl_taxi_driver.rl_agent`: Defines the reinforcement learning agents (`QLearningAgent`, `SarsaAgent`) used for training.
- `rl_taxi_driver.train`: Contains the function to train the agent (`train_agent`) based on the selected reinforcement learning algorithm.
- `rl_taxi_driver.utils`: Provides utility functions for plotting returns (`plot_returns`) and displaying learned policies (`show_policy`).
- `main.py`: Entry point for running the training process and visualizing the trained agent's policy.

## Setup

To run the project, make sure you have Python and Poetry installed. You can install Poetry by following the instructions [here](https://python-poetry.org/docs/#installation).

After installing Poetry, navigate to the project directory and install the dependencies:
```bash
poetry install
```

## Usage

You can run the training process by executing the `main.py` script. The `main` function in `main.py` takes the following parameters:

- `agent`: The reinforcement learning agent to use for training.
- `env`: The environment to train the agent on (Taxi-v3 in this case).
- `n_episodes`: The number of episodes to train the agent.
- `file_name`: The file name for saving the learning curve plot.
```bash
poetry run python main.py
```

## Experiment Tracking

The project integrates with MLflow for experiment tracking. Make sure you have MLflow installed and running. You can set the MLflow tracking URI and experiment name in the `main.py` script. The default values are:

- Tracking URI: http://localhost:8088
- Experiment name: Taxi-Env-RL

To set up MLflow, you can follow the instructions on the [official MLflow documentation](https://www.mlflow.org/docs/latest/index.html).

Once MLflow is set up, you can run the training process as mentioned above. MLflow will track your experiments, including parameters, metrics, and artifacts such as model parameters and learning curve plots.

You can view and manage your experiments using the MLflow UI. By default, MLflow UI is accessible at `http://localhost:8088`. You can navigate to this URL in your web browser to explore experiment results and compare different runs.
