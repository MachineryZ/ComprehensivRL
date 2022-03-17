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

    def run_episode(gamma, epsilon, render=False):
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
            seed = np.random.random()
            if seed < (1 - epsilon):
                obs, reward, done, _ = env.step(int(policy[prev_obs]))
            else:
                obs, reward, done, _ = env.step(env.action_space.sample())
            total_reward += gamma ** step_idx * reward 
            episode.append([prev_obs, policy[prev_obs], reward])
            prev_obs = obs
            step_idx += 1
            if done:
                break
        return episode, total_reward

    def montecarlo_iteration(gamma, max_iteration):
        for _ in range(max_iteration):
            episode, _ = run_episode(gamma, epsilon=epsilon, render=False)
            l = len(episode)
            g = 0
            for i in reversed(range(l)):
                s = episode[i][0]
                a = episode[i][1]
                r = episode[i][2]
                g = gamma * g + r
                n[s][a] = n[s][a] + 1
                q[s][a] = q[s][a] + (g - q[s][a])/n[s][a]
            for s in range(num_states):
                policy[s] = np.argmax(q[s])
    
    def evaluate_policy(gamma, n):
        scores = []
        for i in range(n):
            _, reward = run_episode(gamma, epsilon=0.0, render=False)
            scores.append(reward)
        return np.mean(scores)
    
    # Init env
    env_name = 'FrozenLake-v1'
    env = gym.make(env_name)

    num_states = env.observation_space.n
    num_action = env.action_space.n
    n = np.zeros((num_states, num_action))
    q = np.zeros((num_states, num_action))
    v = np.zeros(num_action)
    gamma = 1.0
    epsilon = 0.9
    policy = np.random.randint(0, num_action, size=(num_states))
    montecarlo_iteration(gamma=gamma, max_iteration=100000)
    scores = evaluate_policy(gamma=gamma, n=10)
    print(f"MC iteration average scores are: {scores}")


def DynamicProgramming():
    pass

# import gym
# import numpy as np
# from itertools import product


# def epsilon_action(a,env,epsilon = 0.1):
#     """ 
#     Return the action most of the time but in 1-epsiolon of the cases a random action within the env.env.action_space is returned
#     Return: action
#     """
#     rand_number = np.random.random()
#     if rand_number < (1-epsilon):
#         return a
#     else:
#         return env.action_space.sample()

# def play_a_game(env,policy,epsilon=0.1):
#     """ 
#     Returns the lived S,A,R of an episode (as a list of a list). In monte carlo the path is epsion-greed partly random.
#     Args: env: Gym enviroment policy: the current policy
#     Return: List of all states with S,A,R for each.
#     """
#     env.reset()
#     finished = False
#     episode_sar = []
#     while not finished:
#         current_s= env.env.s
#         action = epsilon_action(policy[current_s],env,epsilon=epsilon)
#         new_s, reward, finished, _info =  env.step(action)
#         episode_sar.append([current_s,action,reward])
#     #episode_sar.append([new_s,None,reward])
#     return episode_sar

# def sar_to_sag(sar_list,GAMMA=0.9):
#     """ 
#     The gain G in Monte Carlo is caluclates by means of the reward for each state and a discount factor Gamma for earlier episondes.
#     Careful: SAR list needs to be reversed for correct calculation of G
#     Args: sar_list: List of S,A,R values of this episode. Gamma: discount factor for future episodes
#     Return: List of all states with S,A,G for each state visited.
#     """
#     G = 0
#     state_action_gain = []
#     for s,a,r in reversed(sar_list):
#         G = r + GAMMA*G
#         state_action_gain.append([s,a,G])
#     return reversed(state_action_gain)


# def monte_carlo(env, episodes=10000, epsilon=0.1):
#     """ 
#     Function for generating a policy the monte carlo way: Play a lot, find the optimal policy this way
#     Args: env: the open ai gym enviroment object
#     Return: policy: the "optimal" policy V: the value table for each s (optional)
#     """
#     #create a random policy
#     policy = {j:np.random.choice(env.action_space.n) for j in range(env.observation_space.n)} 
#     #Gain or return is cummulative rewards over the entiere episode g(t) = r(t+1) + gamma*G(t+1)
#     G = 0
#     #Q function is essential for the policy update
#     Q = {j:{i:0 for i in range(env.action_space.n)} for j in range(env.observation_space.n)} 
#     #The s,a pairs of the Q function are updated using mean of returns of each episode. So returns need to be collected
#     returns = {(s,a):[] for s,a in product(range(env.observation_space.n),range(env.action_space.n))}

#     for ii in range(episodes):
#         seen_state_action_pairs = set()
#         #play a game and convert S,A,R to S,A,G
#         episode_sag = sar_to_sag(play_a_game(env,policy,epsilon=epsilon if ii > 1000 else 1))        
#         #Use S,A,G to update Q (first-visit method), retruns and seen_state_action_paris
#         for s,a,G in episode_sag:
#             sa = (s, a)
#             if sa not in seen_state_action_pairs:
#                 returns[sa].append(G)
#                 Q[s][a] = np.mean(returns[sa])
#                 seen_state_action_pairs.add(sa)
#         # calculate new policy p[s] = argmax[a]{Q(s,a)}
#         for s in policy.keys():
#             policy[s] = max(Q[s],key=Q[s].get)

#     #optional: create V[s]
#     V = {s:max(list(Q[s].values())) for s in policy.keys()}

#     return policy, V

# def test_policy(env,policy):
#     env.reset()
#     finished = False
#     while not finished:
#         _new_s, _reward, finished, _info =  env.step(policy[env.env.s])
#         # env.render()
#         if finished:
#             break

# def main():
#     #env = gym.make('FrozenLake8x8-v0')
#     env = gym.make('FrozenLake-v1')
#     # env.render()
#     policy, V = monte_carlo(env,episodes=10000,epsilon=0.1)   
#     print(policy)
#     test_policy(env,policy)

# main()

if __name__ == '__main__':
    MonteCarloMethod()
    pass

# import gym
# env = gym.make('FrozenLake-v1')
# for i in range(10000):
#     env.reset()
#     finished = False
#     while not finished:
#         s, r, finished, _info = env.step(env.action_space.sample())
#         if r > 0:
#             print(r)
