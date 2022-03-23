# import gym
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.distributions import Categorical

# """
# Score function for log probability:
# nabla theta = alpha r (partial log p(a|s) / (partial theta))

# probs = policy_network(sate)
# m = Categorical(probs)
# action = m.sample()
# next_state, reward = env.step(action)
# loss = -m.log_prob(action) * reward
# """

# """
# PPO:


# Importance Sampling:


# """
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         bias: float = True,
#     ):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
#         self.fc3 = nn.Linear(hidden_dim, output_dim, bias=bias)
#         self.gelu = nn.GELU()
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x: torch.Tensor):
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.fc2(x)
#         x = self.gelu(x)
#         x = self.fc3(x)
#         x = self.softmax(x)
#         return x

# class ProximalPolicyOptimization(object):
#     def __init__(
#         self,
#         env,
#         state_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         train_epochs: int,
#         test_epochs: int,
#         lr: float,
#         max_iteration_length: int,
#         seed: int,
#         gamma: float,
#         update_frequency: int,
#         epsilon: float,
#         n_epochs: int,
#         gae_lambda: float,
#     ):
#         self.env = env
#         self.train_epochs = train_epochs
#         self.test_epochs = test_epochs
#         self.lr = lr
#         self.max_iteration_length = max_iteration_length
#         self.seed = seed
#         self.gamma = gamma
#         self.update_frequency = update_frequency
#         self.epsilon = epsilon
#         self.n_epochs = n_epochs
#         self.gae_lambda = gae_lambda
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.actor_net = MLP(state_dim, hidden_dim, output_dim).to(self.device)
#         self.critic_net = MLP(state_dim, hidden_dim, output_dim).to(self.device)

#         # Set seed
#         torch.manual_seed(seed)
#         self.env.seed(seed)

#         self.state_list = []
#         self.actor_action_list = []
#         self.critic_action_list = []
#         self.reward_list = []
#         self.log_probs_list = []
#         self.done_list = []


#     def train(self):
#         epoch_reward_list = []
#         for i in range(self.train_epochs):
#             state = self.env.reset()
#             epoch_reward = 0.0
#             for _ in range(self.max_iteration_length):
#                 action = self.select_action(state)
#                 state, reward, finished, info = self.env.step(action)
#                 epoch_reward += reward
#                 if finished:
#                     break
#             if len(epoch_reward_list) == 0:
#                 epoch_reward_list.append(epoch_reward)
#             else:
#                 epoch_reward_list.append(epoch_reward_list[-1] * 0.1 + 0.9 * epoch_reward)
#             if i % self.update_frequency == 0:
#                     self.update_policy()
#         return

#     def test(self):
#         epoch_reward_list = []
#         for i in range(self.test_epochs):
#             state = self.env.reset()
#             epoch_reward = 0.0
#             for _ in range(self.max_iteration_length):
#                 actor_action, log_probs, critic_action = self.select_action(state)
#                 state, reward, finished, info = self.env.step(actor_action)
#                 epoch_reward += reward
#                 if finished:
#                     break
#             epoch_reward_list.append(reward)
#         self.plot(epoch_reward_list, mode="test")
#         return

#     def select_action(self, state):
#         state = torch.tensor([state], dtype=torch.float).to(self.device)
#         # Actor Net:
#         actor_output = self.actor_net(state) # get actor net output
#         actor_probs = Categorical(actor_output) # get distribution
#         actor_action = actor_probs.sample() # sample a action (in tensor)
#         log_probs = torch.squeeze(actor_probs.log_prob(actor_action)).item() # get log probability
#         actor_action = torch.squeeze(actor_action).item() # change tensor into int
        
#         # Critic Net:
#         critic_action = self.critic_net(actor_output) # get critic net output
#         critic_action = torch.squeeze(critic_action).item() # change tensor into int

#         return actor_action, log_probs, critic_action

#     def update(self):

#         for _ in range(self.n_epochs):
#             self.actor_action_list = np.array(self.actor_action_list)
#             self.critic_action_list = np.array(self.critic_action_list)
#             self.state_list = np.array(self.state_list)
#             self.log_probs_list = np.array(self.log_probs_list)
#             self.done_list = np.array(self.done_list)
#             self.reward_list = np.array(self.reward_list)
#             advantage = np.zeros(len(self.reward_list), dtype=np.float32)
#             for t in range(len(advantage) - 1):
#                 discount = 1
#                 a_t = 0
#                 for k in range(t, len(advantage) - 1):
#                     a_t += discount * (self.reward_list[k] + self.gamma * self.log_probs_list[k+1] * \
#                                         (1 - int(self.done_list[k]) - self.critic_action_list[k]))
#                     discount = discount * self.gamma * self.gae_lambda
#                 advantage[t] = a_t
#             advantage = torch.tensor(advantage).to(self.device)
#             # Backpropogation
#             for batch


#         self.actor_action_list = []
#         self.critic_action_list = []
#         self.reward_list = []
#         self.done_list = []
#         self.log_probs_list = []
#         self.state_list = []
#         return



#     def plot(self, reward_list, mode):
#         if mode == 'train':
#             plt.figure()
#             plt.title("train learning curve of policy gradient for cartpole")
#             plt.plot(reward_list)
#             plt.savefig("chapter_5_ppo_train.png")
#         elif mode == 'test':
#             plt.figure()
#             plt.title("test learning curve of policy gradient for cartpole")
#             plt.plot(reward_list)
#             plt.savefig("chapter_5_ppo_test.png")

#     def pipeline(self):
#         pass

# if __name__ == "__main__":
#     env = gym.make("CartPole-v1")
#     ppo = ProximalPolicyOptimization(
#         env=env,
#     )
#     ppo.pipeline()


import numpy as np
import gym
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]
        return np.array(self.states),np.array(self.actions),np.array(self.probs),\
                np.array(self.vals),np.array(self.rewards),np.array(self.dones),batches
                
    def push(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

import torch.nn as nn
from torch.distributions.categorical import Categorical
class Actor(nn.Module):
    def __init__(self,state_dim, action_dim,
            hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
        )
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        value = self.critic(state)
        return value

import os
import numpy as np
import torch 
import torch.optim as optim
class PPO:
    def __init__(self, state_dim, action_dim,cfg):
        self.gamma = cfg.gamma
        # self.continuous = cfg.continuous 
        self.continuous = False
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(state_dim, action_dim,cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_dim,cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        if self.continuous:
            action = torch.tanh(action)
        else:
            action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            ### compute advantage ###
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()  
    def save(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
    def load(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint)) 
        self.critic.load_state_dict(torch.load(critic_checkpoint))  

class PPOConfig:
    def __init__(self) -> None:
        self.env = 'CartPole-v1'
        self.algo = 'PPO'
        self.result_path = curr_path+"/results/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/results/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 200 # max training episodes
        self.test_eps = 50
        self.batch_size = 5
        self.gamma=0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda=0.95
        self.policy_clip=0.2
        self.hidden_dim = 256
        self.update_fre = 20 # frequency of agent update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check gpu
import matplotlib.pyplot as plt
import seaborn as sns
def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()
def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env)  
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim,action_dim,cfg)
    return env,agent
import gym
import torch
import datetime
def save_results(rewards, ma_rewards, tag='train', path='./results'):
    ''' 保存奖励
    '''
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('结果保存完毕！')


def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
def train(cfg,env,agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards= []
    ma_rewards = [] # moving average rewards
    running_steps = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            running_steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if running_steps % cfg.update_fre == 0:
                agent.update()
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%10==0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
    print('Complete training！')
    return rewards,ma_rewards

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards= []
    ma_rewards = [] # moving average rewards
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%10==0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
    print('Complete evaling！')
    return rewards,ma_rewards
import sys
from pathlib import Path
curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # add current terminal path to sys.path
if __name__ == '__main__':
    cfg  = PPOConfig()
    # train
    env,agent = env_agent_config(cfg,seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train",
                 algo=cfg.algo, path=cfg.result_path)
    # eval
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)