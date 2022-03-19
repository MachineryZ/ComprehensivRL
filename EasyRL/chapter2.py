# Chapter 2
# Mainly focus on MDP 
# Value function
# Q function
# Value iteration
# Policy iteration
"""
Realize policy iteration code
Realize value iteration code
Reference:
https://github.com/cuhkrlcourse/RLexample/blob/master/MDP/frozenlake_policy_iteration.py
https://github.com/cuhkrlcourse/RLexample/blob/master/MDP/frozenlake_value_iteration.py
"""



"""
Policy iteration Algorithms:
"""

import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

def frozen_lake_policy_iteration():

    def compute_policy_v(policy, gamma):
        """Compute the value function for given policy

        Args:
            policy (np.ndarray): given a policy
            gamma (float): discount factor

        Returns:
            np.ndarray: value function

        Notice:
        env.env.P is a dict: observation_space -> list[(possible, next_sate, reward, info)]
        which means, if you want to access to next 
        """
        eps = 1e-10
        v = np.zeros(env.observation_space.n)
        while True: # iteratively compute the value function until convergence
            prev_v = np.copy(v)
            for s in range(env.observation_space.n):
                a = policy[s]
                v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]])
            if np.sum((np.fabs(prev_v - v))) <= eps:
                break
        return v
    
    def extract_policy(v, gamma):
        """Calculate an optimal policy given a value function

        Args:
            v (np.ndarray): value function
            gamma (_type_): discount factor

        Returns:
            np.ndarray: optimal policy
        """
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            q_sa = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
            policy[s] = np.argmax(q_sa)
        return policy

    def evaluate_policy(policy, gamma, n):
        """Evaluate how a policy performes

        Args:
            policy (np.ndarray): policy
            gamma (float): discount factor
            n (int): iteration times

        Returns:
            float: scores
        """
        scores = [run_episode(policy=policy, gamma=gamma, render=False) for _ in range(n)]
        return np.mean(scores)

    def run_episode(policy, gamma, render=False):
        """Run an episode and return a total reward given a policy

        Args:
            policy (np.ndarray): policy
            gamma (float): discount factor
            rendor (bool): whether rendor or not

        Returns:
            float: total_reward
        """
        obs = env.reset()
        total_reward = 0.0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(int(policy[obs]))
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward
    
    # create frozen lake environment 
    env_name = 'FrozenLake-v1'
    env = gym.make(env_name)

    # Get Optimal Policy
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n)) # initial a random policy
    max_iteration = 1000000
    gamma = 0.99
    n = 100
    for i in range(max_iteration):
        old_value_v = compute_policy_v(policy=policy, gamma=gamma)
        new_policy = extract_policy(v=old_value_v, gamma=gamma)
        if (np.all(policy == new_policy)):
            print("Policy-Iteration converged at step %d." %(i+1))
            break
        policy = new_policy
    scores = evaluate_policy(policy, gamma, n)
    print("Policy iteration average scores = ", np.mean(scores))
    print("Policy iteration final policy = ", policy)
    return 

"""
Value iteration method
"""

def frozen_lake_value_iteration():

    def extract_policy(v, gamma):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            q_sa = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for next_sr in env.env.P[s][a]:
                    p, s_, r, _ = next_sr
                    q_sa[a] += (p * (r + gamma * v[s_]))
            policy[s] = np.argmax(q_sa)
        return policy

    def value_iteration(gamma):
        v = np.zeros(env.observation_space.n)
        max_iterations = 100000
        eps = 1e-10
        for i in range(max_iterations):
            prev_v = np.copy(v)
            for s in range(env.observation_space.n):
                q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.action_space.n)]
                v[s] = max(q_sa)
            if (np.sum(np.fabs(prev_v - v)) <= eps):
                print("Value iteration converged at iteration {}".format(i))
                break
        return v
    
    def evaluate_policy(policy, gamma, n):
        scores = [run_episode(policy, gamma, render=False) for _ in range(n)]
        return np.mean(scores)

    def run_episode(policy, gamma, render):
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(policy[obs])
            total_reward += gamma ** step_idx * reward
            step_idx += 1
            if done is True:
                break

        return total_reward

    # create frozen lake environment
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)

    # initialize value-function
    gamma = 0.99
    optimal_v = value_iteration(gamma)
    policy = extract_policy(optimal_v, gamma)
    policy_score = evaluate_policy(policy, gamma, n=100)
    print(f"Value iteration scores are {policy_score}") 
    print(f"Value iteration final policy ", policy)
    return       


if __name__ == '__main__':
    frozen_lake_policy_iteration()
    frozen_lake_value_iteration()