import gym
import numpy as np

def MonteCarloMethod(
    method: str = None,
):
    """
    Monte Carlo Method (value function version)

    1. Initial value function, number function: v, n
    2. generate an episode:
        1. episode: (s0, a0, r1, s1, a1, r2, s2, ..., an-1, rn, sn)
        2. reverse traverse: r = rn
            for i = n:1:
                n[si] = n[si] + 1
                v[si] = v[si] + (ri + gamma * r - v[si])/n[si]
                r = ri + gamma r
        3. update your policy:
            p[si] = argmax(v[si])
    """

    def run_episode(gamma, render=False):
        """
        Generate episode

        Args:
            gamma (float): discount factor
            render (bool, optional): whether to render or not. Defaults to False.

        Returns:
            list: [si, ai, ri] episode list,
            float: total reward
        """
        prev_obs = env.reset()
        step_idx = 0
        episode = []
        total_reward = 0.0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(int(policy[prev_obs]))
            total_reward += gamma ** step_idx * reward 
            episode.append([prev_obs, policy[prev_obs], reward])
            prev_obs = obs
            step_idx += 1
            if done:
                break
        return episode, total_reward

    def montecarlo_iteration(gamma, max_iteration):
        for _ in range(max_iteration):
            episode, _ = run_episode(gamma, render=False)
            l = len(episode)
            g = 0
            for i in reversed(range(l)):
                s = episode[i][0]
                r = episode[i][2]
                g = gamma * g + r
                n[s] = n[s] + 1
                v[s] = v[s] + (g - v[s])/n[s]
            for s in range(num_states):
                policy[i] = np.argmax(v[s])
    
    def evaluate_policy(gamma, n):
        scores = []
        for i in range(n):
            _, reward = run_episode(gamma, False)
            scores.append(reward)
        return np.mean(scores)
    
    # Init env
    env_name = 'FrozenLake-v1'
    env = gym.make(env_name)

    num_states = env.observation_space.n
    num_action = env.action_space.n
    n = np.zeros(num_states)
    v = np.zeros(num_states) - 1
    gamma = 0.9
    policy = np.random.choice(num_action, size=(num_states))
    montecarlo_iteration(gamma=gamma, max_iteration=10000)
    scores = evaluate_policy(gamma=gamma, n=10)
    print(f"MC iteration average scores are: {scores}")
    print(f"MC iteration final policy ", policy)


def DynamicProgramming():
    pass

    

if __name__ == '__main__':
    MonteCarloMethod()