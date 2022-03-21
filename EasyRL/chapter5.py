import gym
import torch
import torch.nn as nn

class ProximalPolicyOptimization(object):
    def __init__(
        self,
        env,
    ):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def plot(self):
        pass

    def pipeline(self):
        pass

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    ppo = ProximalPolicyOptimization(
        env=env,
    )
    ppo.pipeline()