import gymnasium as gym
from rl_taxi_driver.q_learning_agent import QLearningAgent
from rl_taxi_driver.train import train_agent
from rl_taxi_driver.utils import plot_returns, show_policy


def main(agent: QLearningAgent, env: gym.Env, n_episodes: int):

    returns = train_agent(agent, env, n_episodes)
    plot_returns(returns, file_name="q_learning_curve.png")

    env = gym.make("Taxi-v3", render_mode="human")
    show_policy(trained_agent=agent, env=env)


if __name__ == "__main__":
    n_episodes = 50000
    initial_epsilon = 1
    final_epsilon = 0
    epsilon_decay = (initial_epsilon - final_epsilon) / (n_episodes / 2)
    alpha = 0.5
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(
        env,
        alpha,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
    )
    main(agent, env, n_episodes)