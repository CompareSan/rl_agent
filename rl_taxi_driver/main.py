import gymnasium as gym
import mlflow

from rl_taxi_driver.rl_agent import (
    QLearningAgent,
    RLAgent,
    SarsaAgent,
)
from rl_taxi_driver.train import train_agent
from rl_taxi_driver.utils import (
    plot_returns,
    show_policy,
)


def main(agent: RLAgent, env: gym.Env, n_episodes: int, file_name: str):

    returns = train_agent(agent, env, n_episodes)
    plot_returns(returns, file_name=file_name)

    env = gym.make("Taxi-v3", render_mode="human")
    show_policy(trained_agent=agent, env=env)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:8088")
    mlflow.set_experiment("Taxi-Env-RL")
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
    main(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        file_name=f"{type(agent).__name__}_learning_curve.png",
    )
