import gymnasium as gym
import mlflow
import numpy as np

from rl_taxi_driver.rl_agent import RLAgent


def train_agent(
    agent: RLAgent,
    env: gym.Env,
    n_episodes: int = 1000,
    last_n: int = 100,
):
    return_for_each_episode = []
    length_for_each_episode = []
    best_avg_return = -np.inf
    with mlflow.start_run():
        params = {
            "type": type(agent).__name__,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "initial_epsilon": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "final_epsilon": agent.final_epsilon,
            "n_episodes": n_episodes,
        }
        mlflow.log_params(params)
        for i in range(n_episodes):
            done = False
            cumulative_return = 0
            episode_lenght = 0
            state, _ = env.reset()

            while not done:
                action = agent.take_action(state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(
                    state,
                    action,
                    reward,
                    terminated,
                    new_state,
                )
                state = new_state
                episode_lenght += 1
                cumulative_return += reward
                done = terminated or truncated

            agent.decay_epsilon()
            return_for_each_episode.append(cumulative_return)
            length_for_each_episode.append(episode_lenght)

            if i >= last_n:
                avg_return = np.mean(return_for_each_episode[i - last_n : i])
                if avg_return > best_avg_return:
                    best_avg_return = avg_return

            if i % last_n == 0 and i > 0:
                print(f"Episode {i}: best average return = {best_avg_return}")

        mlflow.log_metric("best_return", best_avg_return)
    return return_for_each_episode
