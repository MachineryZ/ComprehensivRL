import gym
from soupsieve import select
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from typing import List
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=True) # only predict the prob to left (cartpole env)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.softmax(x) 
        return x

class PolicyGradient(object):
    def __init__(
        self,
        env,
        state_dim: int,
        action_dim: int,
        hidden_dim: List,
        train_epochs: int,
        test_epochs: int,
        lr: float,
        gamma: float,
        max_iteration_length: int, 
        seed: int,
    ):
        """
        Policy Gradient:

        theta = theta + lr * nabla(log(pi_theta(s_t, a_t) * R )

        Args:
            env (_type_): 
            state_dim (int): 
            action_dim (int): 
            hidden_dim (List): 
            train_epochs (int): 
            test_epochs (int): 
            lr (float): 
            gamma (float): 
            max_iteration_length (int): 
        """
        self.env = env
        self.env.seed(seed)
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.lr = lr
        self.gamma = gamma
        self.max_iteration_length = max_iteration_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        self.optimizer = torch.optim.Adam(
            params=self.policy_net.parameters(),
            lr=self.lr,
        )
        self.policy_net = self.policy_net.to(self.device)

    def select_action(self, state, epoch_action_list=None):
        # Transfer numpy array into torch.tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        if epoch_action_list is not None:
            epoch_action_list.append(m.log_prob(action))
        return action.item()

    def train(self):
        ma_total_reward_list = []
        total_reward_list = []
        for i in range(self.train_epochs):
            state = self.env.reset()
            epoch_reward_list = []
            epoch_action_list = []
            total_reward = 0.0
            for _ in range(self.max_iteration_length):
                # Count distribution of action and random sample one:
                action = self.select_action(state, epoch_action_list)

                # interact with environment
                state, reward, finished, info = self.env.step(action)
                epoch_reward_list.append(reward)
                total_reward += reward
                if finished:
                    break
            total_reward_list.append(total_reward)
            if len(ma_total_reward_list) == 0:
                ma_total_reward_list.append(total_reward)
            else: # moving average
                ma_total_reward_list.append(ma_total_reward_list[-1] * 0.05 + (1 - 0.05) * total_reward)
            self.update_policy(epoch_reward_list, epoch_action_list)
        self.plot(ma_total_reward_list, mode="train")
        # self.plot(total_reward_list, mode="train")

    def update_policy(self, 
            epoch_reward_list: List,
            epoch_action_list: List,
        ):
        R = .0
        policy_loss = []
        returns = []
        for r in epoch_reward_list[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        for log_prob, R in zip(epoch_action_list, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum().to(self.device)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        return

    def test(self):
        test_reward_list = []
        for i in range(self.test_epochs):
            state = self.env.reset()
            finished = False
            epoch_reward = .0
            for i in range(self.max_iteration_length):
                action = self.select_action(state)
                state, reward, finished, info = self.env.step(action)
                epoch_reward += reward
                if finished:
                    break
            test_reward_list.append(epoch_reward)
        self.plot(test_reward_list, mode="test")
        return

    def plot(self, reward_list: List, mode: str):
        if mode == 'train':
            plt.figure()
            plt.title("train learning curve of policy gradient for cartpole")
            plt.plot(reward_list)
            plt.savefig("chapter_4_policy_gradient_train.png")
        elif mode == 'test':
            plt.figure()
            plt.title("test learning curve of policy gradient for cartpole")
            plt.plot(reward_list)
            plt.savefig("chapter_4_policy_gradient_test.png")

    def pipeline(self):
        self.train()
        self.test()
    
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    pg = PolicyGradient(
        env=env,
        state_dim=4,
        action_dim=2,
        hidden_dim=32,
        train_epochs=320,
        test_epochs=100,
        lr=0.005,
        gamma=0.99,
        max_iteration_length=1000,
        seed=7777,
    )
    pg.pipeline()

# https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

# import argparse
# import gym
# import numpy as np
# from itertools import count

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical


# parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor (default: 0.99)')
# parser.add_argument('--seed', type=int, default=543, metavar='N',
#                     help='random seed (default: 543)')
# parser.add_argument('--render', action='store_true',
#                     help='render the environment')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='interval between training status logs (default: 10)')
# args = parser.parse_args()


# env = gym.make('CartPole-v1')
# env.seed(args.seed)
# torch.manual_seed(args.seed)


# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.dropout = nn.Dropout(p=0.6)
#         self.affine2 = nn.Linear(128, 2)

#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         x = self.affine1(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         action_scores = self.affine2(x)
#         return F.softmax(action_scores, dim=1)


# policy = Policy()
# optimizer = optim.Adam(policy.parameters(), lr=1e-2)
# eps = np.finfo(np.float32).eps.item()


# def select_action(state):
#     state = torch.from_numpy(state).float().unsqueeze(0)
#     probs = policy(state)
#     m = Categorical(probs)
#     action = m.sample()
#     policy.saved_log_probs.append(m.log_prob(action))
#     return action.item()


# def finish_episode():
#     R = 0
#     policy_loss = []
#     returns = []
#     for r in policy.rewards[::-1]:
#         R = r + args.gamma * R
#         returns.insert(0, R)
#     returns = torch.tensor(returns)
#     returns = (returns - returns.mean()) / (returns.std() + eps)
#     for log_prob, R in zip(policy.saved_log_probs, returns):
#         policy_loss.append(-log_prob * R)
#     optimizer.zero_grad()
#     policy_loss = torch.cat(policy_loss).sum()
#     policy_loss.backward()
#     optimizer.step()
#     del policy.rewards[:]
#     del policy.saved_log_probs[:]


# def main():
#     running_reward = 10
#     for i_episode in count(1):
#         state, ep_reward = env.reset(), 0
#         for t in range(1, 10000):  # Don't infinite loop while learning
#             action = select_action(state)
#             state, reward, done, _ = env.step(action)
#             if args.render:
#                 env.render()
#             policy.rewards.append(reward)
#             ep_reward += reward
#             if done:
#                 break

#         running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
#         finish_episode()
#         if i_episode % args.log_interval == 0:
#             print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
#                   i_episode, ep_reward, running_reward))
#         if running_reward > env.spec.reward_threshold:
#             print("Solved! Running reward is now {} and "
#                   "the last episode runs to {} time steps!".format(running_reward, t))
#             break


# if __name__ == '__main__':
#     main()