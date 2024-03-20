import gymnasium as gym
import pytest

from rl_taxi_driver.q_learning_agent import QLearningAgent
from rl_taxi_driver.train import train_agent


@pytest.fixture()
def trained_agent_definition():
    env = gym.make("Taxi-v3")
    n_episodes = 1
    initial_epsilon = 1
    final_epsilon = 0
    epsilon_decay = (initial_epsilon - final_epsilon) / (n_episodes / 2)
    alpha = 0.5
    agent = QLearningAgent(
        env,
        alpha,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
    )
    returns = train_agent(agent, env, n_episodes)
    return returns


def test_returns_is_a_list(trained_agent_definition):
    assert isinstance(trained_agent_definition, list)


def test_len_returns(trained_agent_definition):
    assert len(trained_agent_definition) == 1
