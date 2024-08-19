import numpy as np
from tqdm.notebook import tqdm
import gymnasium as gym

# This class defines the Monte-Carlo agent
class MC_agent(object):
    
    def generate_episode(self, env, policy):
        episode = []
        state, _ = env.reset()
        done = False
        action = np.random.choice(list(range(env.action_space.n)), p=policy[state])
        episode.append((0, state, action))  # Initial reward is 0
        
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            episode.append((reward, next_state, action))
            state = next_state
            if not done:
                action = np.random.choice(list(range(env.action_space.n)), p=policy[state])
        
        total_reward = sum([step[0] for step in episode])
        return episode, total_reward
        
    def solve(self, env, total_episodes=1000, epsilon='1/cbrt(n_episodes)'):
        """
        Solve a given environment using Monte Carlo learning
        input: env {Gymnasium environment} -- Environment to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of rewards for each episode 
        """
        # Initialization
        state_size = env.observation_space.n
        action_size = env.action_space.n
        Q = np.random.rand(state_size, action_size)
        V = np.zeros(state_size)
        
        # Generate a random soft policy
        policy = np.random.rand(state_size, action_size)
        policy /= np.sum(policy, axis=1)[:,None] 
        policy = np.ones((state_size, action_size)) * 0.25
        
        values = []
        total_rewards = []
        
        counts = dict()
        
        for n_episodes in tqdm(range(total_episodes), unit="episode", leave=False):
            
            episode, total_reward = self.generate_episode(env, policy)
            total_rewards.append(total_reward)
            
            G = 0
            if isinstance(epsilon, str) and epsilon == '1/cbrt(n_episodes)':
                E = 1/np.cbrt(n_episodes+1)
            elif isinstance(epsilon, str) and epsilon == '1/sqrt(n_episodes)':
                E = 1/np.sqrt(n_episodes+1)
            elif isinstance(epsilon, str) and epsilon == '1/n_episodes':
                E = 1/(n_episodes+1)
            elif isinstance(epsilon, float):
                E = epsilon
            
            for t in range(len(episode)-2, -1, -1):
                G = G + episode[t+1][0]  # Adding immediate reward (no discounting in Frozen Lake)
                s_t = episode[t][1] 
                a_t = episode[t][2]
                
                # On-policy first visit MC condition
                if not any(episode[prior_t][1] == s_t and episode[prior_t][2] == a_t for prior_t in range(0, t)):
                    if (s_t, a_t) not in counts:
                        counts[(s_t, a_t)] = 1
                    else:
                        counts[(s_t, a_t)] += 1
                    
                    learning_rate = 1 / counts[(s_t, a_t)]
                    Q[s_t, a_t] = Q[s_t, a_t] + learning_rate * (G - Q[s_t, a_t])
                    
                    # e-greedy policy update
                    a_opt = np.argmax(Q[s_t, :])
                    for a in range(action_size):
                        if a == a_opt:
                            policy[s_t][a] = 1 - E + E/action_size
                        else:
                            policy[s_t][a] = E/action_size
            
            values.append(np.sum(policy * Q, axis=1))

        return policy, values, total_rewards

# Example usage with FrozenLake-v1 environment
env = gym.make("FrozenLake-v1", is_slippery=True)
mc_agent = MC_agent()
policy, values, total_rewards = mc_agent.solve(env)
