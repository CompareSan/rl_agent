import gymnasium as gym
import numpy as np

from rl_taxi_driver.q_learning_agent import QLearningAgent


def train_agent(
    agent: QLearningAgent,
    env: gym.Env,
    n_episodes: int = 1000,
    last_n: int = 100,
):
    return_for_each_episode = []
    length_for_each_episode = []
    best_avg_return = -np.inf
    for i in range(n_episodes):
        done = False
        cumulative_return = 0
        episode_lenght = 0
        state, _ = env.reset()

        while not done:
            action = agent.take_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            agent.update_q_function(
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

    return return_for_each_episode
