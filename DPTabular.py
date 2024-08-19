import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DP_agent_FrozenLake:
    def __init__(self, gamma=0.99, threshold=0.0001):
        self.gamma = gamma
        self.threshold = threshold

    def solve(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        V = np.zeros(n_states)
        policy = np.zeros((n_states, n_actions))
        
        delta = self.threshold
        epochs = 0

        while delta >= self.threshold:
            epochs += 1
            delta = 0.0

            for state in range(n_states):
                opt_v, opt_a = float('-inf'), 0

                for action in range(n_actions):
                    v = 0
                    for prob, next_state, reward, done in env.P[state][action]:
                        v += prob * (reward + self.gamma * V[next_state])
                    
                    if v > opt_v:
                        opt_v, opt_a = v, action

                delta = max(delta, np.abs(opt_v - V[state]))
                V[state] = opt_v
                policy[state] = np.eye(n_actions)[opt_a]  # one-hot encode the optimal action

        return policy, V, epochs

def plot_value_function(value_function, map_name):
    size = int(np.sqrt(len(value_function)))  # Assuming square map
    grid = value_function.reshape((size, size))
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, cmap='viridis', cbar=True)
    plt.title(f'Value Function for {map_name}')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

def plot_policy(policy, map_name):
    size = int(np.sqrt(len(policy)))  # Assuming square map
    policy_grid = np.argmax(policy, axis=1).reshape((size, size))
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy_grid, annot=True, cmap='coolwarm', cbar=False, 
                linewidths=0.5, linecolor='black')
    plt.title(f'Policy for {map_name}')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

def generate_random_map(size, p=0.1, seed=None):
    np.random.seed(seed)
    map_ = np.random.choice(['S', 'F', 'H', 'G'], size=(size, size), p=[0.1, 0.8, 0.05, 0.05])
    map_[0, 0] = 'S'
    map_[-1, -1] = 'G'
    return map_

if __name__ == "__main__":
    #map_sizes = [4, 7, 9, 11]
    map_sizes = [7]
    for map_size in map_sizes:
        map_name = f"{map_size}x{map_size}"
        env = gym.make(
            "FrozenLake-v1",
            is_slippery=True,
            render_mode="rgb_array",
            desc=generate_random_map(size=map_size, p=0.1, seed=42)
        )

        agent = DP_agent_FrozenLake(gamma=0.99, threshold=0.0001)
        optimal_policy, value_function, epochs = agent.solve(env)

        print(f"Map size: {map_name}")
        print("Optimal Policy (one-hot encoded):")
        print(optimal_policy)
        print("\nValue Function:")
        print(value_function)
        print("\nEpochs to converge:", epochs)

        plot_value_function(value_function, map_name)
        plot_policy(optimal_policy, map_name)
