from abc import (
    ABC,
    abstractmethod,
)

import gymnasium as gym
import numpy as np


class RLAgent(ABC):

    def __init__(
        self,
        env: gym.Env,
        alpha: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        gamma: float = 0.95,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    @abstractmethod
    def take_action(self, state: float) -> float:
        pass

    @abstractmethod
    def update(
        self,
        state: float,
        action: float,
        reward: float,
        terminated: bool,
        new_state: float,
    ) -> float:
        pass

    @abstractmethod
    def decay_epsilon(self) -> None:
        pass


class QLearningAgent(RLAgent):
    def __init__(
        self,
        env: gym.Env,
        alpha: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        gamma: float = 0.95,
    ) -> None:
        super().__init__(
            env,
            alpha,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            gamma,
        )
        self.q_values: np.ndarray = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def take_action(self, state: float):
        exploit = np.argmax(self.q_values[state])
        explore = self.env.action_space.sample()
        random_number = np.random.uniform()
        if random_number < self.epsilon:
            return explore
        else:
            return exploit

    def update(
        self,
        state: float,
        action: float,
        reward: float,
        terminated: bool,
        new_state: float,
    ):

        if terminated:
            q_s_a = 0

        q_s_a = np.max(self.q_values[new_state])
        self.q_values[state, action] += self.alpha * (reward + self.gamma * q_s_a - self.q_values[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
