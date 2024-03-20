import gymnasium as gym
import pytest

from rl_taxi_driver.q_learning_agent import QLearningAgent


@pytest.fixture()
def agent_definition():
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

    return agent


def test_take_action(agent_definition):
    state = 0
    action = agent_definition.take_action(state)
    assert action in range(agent_definition.env.action_space.n)
