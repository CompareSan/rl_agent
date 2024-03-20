import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_taxi_driver.q_learning_agent import QLearningAgent


def show_policy(trained_agent: QLearningAgent, env: gym.Env):
    done = False
    state, _ = env.reset()
    env.render()
    trained_agent.epsilon = 0

    while not done:
        action = trained_agent.take_action(state)
        new_state, _, terminated, truncated, _ = env.step(action)
        state = new_state

        done = terminated or truncated


def plot_returns(returns, file_name):
    plt.plot(np.arange(len(returns)), returns)
    plt.title("Episode returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(file_name)
    plt.show()
