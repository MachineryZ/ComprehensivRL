from audioop import bias
from cmath import exp
from re import U
from turtle import done

from matplotlib import markers
import gym
import math
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        return x

class ReplayBuff:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DoubleDQN(object):
    def __init__(
        self,
        env_name: str,
        capacity: int,
        train_epochs: int,
        test_epochs: int,
        max_iterations: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        hidden_dim: int,
        lr: float,
        bias: bool,
        seed: int,
        update_epoch: int,
        batch_size: int,
        gamma: float,
    ):
        self.env = gym.make(env_name)
        self.capacity = capacity
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.max_iterations = max_iterations
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_epoch = update_epoch
        self.batch_size = batch_size
        self.gamma = gamma

        torch.manual_seed(seed)
        self.env.seed(seed)
        self.input_dim = input_dim = self.env.observation_space.shape[0]
        self.output_dim = output_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = MLP(input_dim, hidden_dim, output_dim, bias).to(self.device)
        self.target_net = MLP(input_dim, hidden_dim, output_dim, bias).to(self.device)
        # Copy the parameters:
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuff(capacity)

        self.frame_idx = 1

    def get_epsilon(self):
        self.frame_idx += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.frame_idx / self.epsilon_decay)
        return epsilon

    def choose_action(self, state, mode):
        if mode == "train":
            epsilon = self.get_epsilon()
            if random.random() > epsilon:
                with torch.no_grad():
                    state = torch.tensor(np.array([state]), device=self.device, dtype=torch.float32)
                    q_values = self.policy_net(state)
                    action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.output_dim)
            return action
        else:
            with torch.no_grad():
                state = torch.tensor(np.array([state]), device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
            return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        # DQN:
        # q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        # next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        # loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        q_values = self.policy_net(state_batch)
        next_q_values = self.policy_net(next_state_batch)
        q_value = q_values.gather(dim=1, index=action_batch)
        next_target_values = self.target_net(next_state_batch)
        next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward_batch + self.gamma * (next_target_q_value) * (1 - done_batch)
        loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        rewards = []
        ma_rewards = []
        for num_epoch in range(self.train_epochs):
            epoch_reward = 0
            state = self.env.reset()
            for i in range(self.max_iterations):
                action = self.choose_action(state, "train")
                next_state, reward, done, info = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                self.update()
                epoch_reward += reward
                if done:
                    break
            rewards.append(epoch_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * epoch_reward)
            else:
                ma_rewards.append(epoch_reward)
            if num_epoch % self.update_epoch == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.plot(rewards, mode="train", extra="na")
        self.plot(ma_rewards, mode="train", extra="ma")

    def test(self):
        rewards = []
        ma_rewards = []
        for num_epoch in range(self.test_epochs):
            epoch_reward = 0.0
            state = self.env.reset()
            for i in range(self.max_iterations):
                action = self.choose_action(state, "test")
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                epoch_reward += reward
                if done:
                    break
            rewards.append(epoch_reward)
            if ma_rewards:
                ma_rewards.append(ma_rewards[-1] * 0.9 + epoch_reward * 0.1)
            else:
                ma_rewards.append(epoch_reward)
        self.plot(rewards, mode="test", extra="na")
        self.plot(ma_rewards, mode="test", extra="ma")

    def plot(self, reward_list, mode: str, extra = str):
        if mode == 'train':
            plt.figure()
            plt.title(f"train learning curve of policy gradient for cartpole {extra}")
            plt.plot(reward_list)
            plt.savefig(f"chapter_7_DoubleDQN_train {extra}.png")
        elif mode == 'test':
            plt.figure()
            plt.title(f"test learning curve of policy gradient for cartpole {extra}")
            plt.plot(reward_list)
            plt.savefig(f"chapter_7_DoubleDQN_test {extra}.png")

class DuelDqnNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool,
    ):
        super(DuelDqnNet, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            self.gelu,
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            self.gelu,
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def choose_action(self, state: int, epsilon: float):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor([state]).unsqueeze(0)
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.output_dim)
        return action

class DuelDQN(object):
    def __init__(
        self,
        env_name: str,
        capacity: int,
        train_epochs: int,
        test_epochs: int,
        max_iterations: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        hidden_dim: int,
        lr: float,
        bias: bool,
        seed: int,
        update_epoch: int,
        batch_size: int,
        gamma: float,
    ):
        self.env = gym.make(env_name)
        self.capacity = capacity
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.max_iterations = max_iterations
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_epoch = update_epoch
        self.batch_size = batch_size
        self.gamma = gamma

        torch.manual_seed(seed)
        self.env.seed(seed)
        self.input_dim = input_dim = self.env.observation_space.shape[0]
        self.output_dim = output_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelDqnNet(input_dim, hidden_dim, output_dim, bias).to(self.device)
        self.target_net = DuelDqnNet(input_dim, hidden_dim, output_dim, bias).to(self.device)
        # Copy the parameters:
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuff(capacity)

        self.frame_idx = 1

    def get_epsilon(self):
        self.frame_idx += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.frame_idx / self.epsilon_decay)
        return epsilon

    def choose_action(self, state, mode):
        if mode == "train":
            epsilon = self.get_epsilon()
            if random.random() > epsilon:
                with torch.no_grad():
                    state = torch.tensor(np.array([state]), device=self.device, dtype=torch.float32)
                    q_values = self.policy_net(state)
                    action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.output_dim)
            return action
        else:
            with torch.no_grad():
                state = torch.tensor(np.array([state]), device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
            return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        # DQN:
        # q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        # next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        # loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        # Double DQN
        # q_values = self.policy_net(state_batch)
        # next_q_values = self.policy_net(next_state_batch)
        # q_value = q_values.gather(dim=1, index=action_batch)
        # next_target_values = self.target_net(next_state_batch)
        # next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        # q_target = reward_batch + self.gamma * (next_target_q_value) * (1 - done_batch)
        # loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        rewards = []
        ma_rewards = []
        for num_epoch in range(self.train_epochs):
            epoch_reward = 0
            state = self.env.reset()
            for i in range(self.max_iterations):
                action = self.choose_action(state, "train")
                next_state, reward, done, info = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                self.update()
                epoch_reward += reward
                if done:
                    break
            rewards.append(epoch_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * epoch_reward)
            else:
                ma_rewards.append(epoch_reward)
            if num_epoch % self.update_epoch == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.plot(rewards, mode="train", extra="na")
        self.plot(ma_rewards, mode="train", extra="ma")

    def test(self):
        rewards = []
        ma_rewards = []
        for num_epoch in range(self.test_epochs):
            epoch_reward = 0.0
            state = self.env.reset()
            for i in range(self.max_iterations):
                action = self.choose_action(state, "test")
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                epoch_reward += reward
                if done:
                    break
            rewards.append(epoch_reward)
            if ma_rewards:
                ma_rewards.append(ma_rewards[-1] * 0.9 + epoch_reward * 0.1)
            else:
                ma_rewards.append(epoch_reward)
        self.plot(rewards, mode="test", extra="na")
        self.plot(ma_rewards, mode="test", extra="ma")

    def plot(self, reward_list, mode: str, extra = str):
        if mode == 'train':
            plt.figure()
            plt.title(f"train learning curve of policy gradient for cartpole {extra}")
            plt.plot(reward_list)
            plt.savefig(f"chapter_7_DuelingDQN_train {extra}.png")
        elif mode == 'test':
            plt.figure()
            plt.title(f"test learning curve of policy gradient for cartpole {extra}")
            plt.plot(reward_list)
            plt.savefig(f"chapter_7_DuelingDQN_test {extra}.png")

if __name__ == '__main__':
    # engine = DoubleDQN(
    #     env_name="CartPole-v1",
    #     capacity=10000,
    #     train_epochs=500,
    #     test_epochs=100,
    #     max_iterations=500,
    #     epsilon_start=0.9,
    #     epsilon_end=0.01,
    #     epsilon_decay=500,
    #     hidden_dim=256,
    #     gamma=0.95,
    #     lr=0.0001,
    #     bias=True,
    #     seed=2022,
    #     update_epoch=4,
    #     batch_size=64,
    # )
    # engine.train()
    # engine.test()
    engine = DuelDQN(
        env_name="CartPole-v1",
        capacity=10000,
        train_epochs=500,
        test_epochs=100,
        max_iterations=500,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=500,
        hidden_dim=256,
        gamma=0.95,
        lr=0.0001,
        bias=True,
        seed=2022,
        update_epoch=4,
        batch_size=64,
    )
    engine.train()
    engine.test()