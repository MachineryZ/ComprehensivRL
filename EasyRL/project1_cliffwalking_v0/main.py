import stat
import gym
import math
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from pandas_datareader import test
"""
main.py:
    basic interface, hyper parameters
agent.py:
    choose_action 
    update
    use epsilon-greedy to exploration and exploitation
model.py:
    actor and critic nn model
memory.py:
    replay buffer
plot.py:
    visualization
"""

class QLearning(object):
    def __init__(
        self,
        env,
        state_dim: int,
        action_dim: int,
        gamma: float,
        alpha: float,
        train_epochs: int,
        test_epochs: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,

    ):
        super().__init__()
        """
        Q-learning update function:
        Q(S, A) = Q(S, A) + alpha * (R + gamma * max_a Q(S', a) - Q(S, A))
        """
        # Initialize some Parameters
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs

        # Initialize Q-function
        self.q = np.zeros((state_dim, action_dim))

    def run_episode(self):
        prev_state = self.env.reset()
        finished = False
        total_reward = 0.0
        while not finished:
            action = self.epsilon_greedy(prev_state)
            state, reward, finished, info = self.env.step(action)
            self.q[prev_state, action] += self.alpha * (reward + self.gamma * \
                (np.max(self.q[state,:]) - self.q[prev_state, action]))
            prev_state = state
            total_reward += reward
        return total_reward

    def epsilon_greedy(self, state):
        seed = np.random.uniform()
        if seed > 1 - self.epsilon: 
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q[state])
        return action

    def train(self):
        reward_list = []
        for i in range(self.train_epochs):
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1. * i / self.epsilon_decay)
            epoch_reward = self.run_episode()
            reward_list.append(epoch_reward)
        self.plot(reward_list, mode="train")

    def test(self):
        reward_list = []
        for i in range(self.test_epochs):
            state = self.env.reset()
            finished = False
            epoch_reward = 0.0
            while not finished:
                state, reward, finished, info = self.env.step(np.argmax(self.q[state]))
                epoch_reward += reward
            reward_list.append(epoch_reward)
        self.plot(reward_list, mode="test")
        

    def plot(self, reward_list: List, mode: str):
        if mode == 'train':
            plt.figure()
            plt.title("train learning curve of q-learning for cliffwalking-v0")
            plt.plot(reward_list)
            plt.savefig('train.png')
        elif mode == 'test':
            plt.figure()
            plt.title("test learning curve of q-learning for cliffwalking-v0")
            plt.plot(reward_list)
            plt.savefig("test.png")

    def pipeline(self):
        self.train()
        self.test()

    
def main():
    env = gym.make("CliffWalking-v0")
    engine = QLearning(
        env=env,
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        gamma=0.9,
        alpha=0.1,
        train_epochs=400,
        test_epochs=100,
        epsilon_start=0.95,
        epsilon_end=0.01,
        epsilon_decay=300,
    )
    engine.pipeline()

if __name__ == '__main__':
    main()
    