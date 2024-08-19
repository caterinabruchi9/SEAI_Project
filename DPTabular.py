import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

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

def visualize_policy_and_value(env, policy, value_function, title="Policy and Value Function"):
    """ Visualize the optimal policy and value function as a grid. """
    size = int(np.sqrt(env.observation_space.n))  # Assuming a square grid

    # Reshape the policy and value function for visualization
    actions = np.argmax(policy, axis=1).reshape(size, size)
    value_function = value_function.reshape(size, size)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the value function
    sns.heatmap(value_function, annot=True, cmap="YlGnBu", ax=ax[0], cbar=True, square=True)
    ax[0].set_title("Value Function")
    
    # Plot the policy
    action_mapping = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_arrows = np.vectorize(action_mapping.get)(actions)
    sns.heatmap(np.zeros_like(actions), annot=policy_arrows, fmt="", cbar=False, cmap="Blues", ax=ax[1], linewidths=1, linecolor='black')
    ax[1].set_title("Optimal Policy")

    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, is_slippery=True)
    agent = DP_agent_FrozenLake(gamma=0.99, threshold=0.0001)

    optimal_policy, value_function, epochs = agent.solve(env)

    print("Optimal Policy (one-hot encoded):")
    print(optimal_policy)
    print("\nValue Function:")
    print(value_function)
    print("\nEpochs to converge:", epochs)

    # Visualize the policy and value function
    visualize_policy_and_value(env, optimal_policy, value_function, title="Frozen Lake: Policy and Value Function")
