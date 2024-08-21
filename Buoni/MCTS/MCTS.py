import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import math

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, action_size):
        return len(self.children) == action_size

    def best_child(self):
        if self.visits == 0:
            return np.random.choice(self.children)
        # Simply choose the child with the highest average reward
        best_value = float('-inf')
        best_child = None
        for child in self.children:
            if child.visits > 0:
                child_value = child.value / child.visits
                if child_value > best_value:
                    best_value = child_value
                    best_child = child
        return best_child

    def expand(self, action, next_state):
        child_node = MCTSNode(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTS:
    def __init__(self, env, num_simulations=1000, convergence_threshold=0.0001):
        self.env = env
        self.num_simulations = num_simulations
        self.convergence_threshold = convergence_threshold
        self.root = None

        # Tracking performance metrics
        self.avg_rewards = []
        self.reward_variances = []
        self.best_values = []
        self.best_visits = []

    def _simulate(self, env_copy, state):
        done = False
        rewards = []
        while not done:
            action = env_copy.action_space.sample()
            state, reward, done, _ , _ = env_copy.step(action)
            rewards.append(reward)
        return np.sum(rewards)

    def search(self, initial_state):
        self.root = MCTSNode(state=initial_state)
        previous_best_value = float('-inf')

        for i in range(self.num_simulations):
            node = self.root
            env_copy = gym.make(self.env.spec.id)
            state = initial_state
            env_copy.reset()

            # Selection
            while node.is_fully_expanded(self.env.action_space.n) and len(node.children) > 0:
                node = node.best_child()
                action = node.action
                state, _, done, _, _ = env_copy.step(action)
                if done:
                    break

            # Expansion
            if not node.is_fully_expanded(self.env.action_space.n):
                action = self._select_untried_action(node)
                next_state, reward, done, _, _ = env_copy.step(action)
                node = node.expand(action, next_state)

                if done:
                    node.update(reward)
                    continue

            # Simulation
            reward = self._simulate(env_copy, node.state)

            # Backpropagation
            self._backpropagate(node, reward)

            # Record metrics
            best_child = self.root.best_child()
            best_value = best_child.value / best_child.visits if best_child.visits > 0 else float('-inf')
            avg_reward = reward
            self.avg_rewards.append(avg_reward)
            self.best_values.append(best_value)
            self.best_visits.append(best_child.visits if best_child.visits > 0 else 0)

            # Calculate variance of rewards
            if len(self.avg_rewards) > 1:
                rewards_mean = np.mean(self.avg_rewards)
                rewards_variance = np.var(self.avg_rewards)
                self.reward_variances.append(rewards_variance)
            else:
                self.reward_variances.append(0.0)

            # Debug information
            print(f"Iteration {i}: Best Value = {best_value}, Avg Reward = {avg_reward}")

            # Check for convergence
            if abs(best_value - previous_best_value) < self.convergence_threshold:
                print(f"Convergence reached with value: {best_value}")
                break
            previous_best_value = best_value

        best_action = self.root.best_child().action
        print(f"Selected Action: {best_action}")
        return best_action

    def _select_untried_action(self, node):
        tried_actions = [child.action for child in node.children]
        possible_actions = set(range(self.env.action_space.n)) - set(tried_actions)
        if possible_actions:
            return np.random.choice(list(possible_actions))
        else:
            return np.random.choice(range(self.env.action_space.n))  # Choose randomly if all actions are tried

    def _backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def plot_performance(self, N=100, title=None, filename=""):
        plt.figure(figsize=(12, 8))

        # Plot average rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.avg_rewards, label='Average Reward', color='blue')
        plt.xlabel('Simulation Iteration')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per Simulation')

        # Plot moving average of rewards if enough data points
        if len(self.avg_rewards) >= N:
            moving_avg = np.convolve(self.avg_rewards, np.ones(N) / N, mode='valid')
            plt.plot(np.arange(N-1, len(self.avg_rewards)), moving_avg, label='Moving Average', color='orange')
        plt.legend()

        # Plot reward variance
        plt.subplot(1, 2, 2)
        plt.plot(self.reward_variances, label='Reward Variance', color='red')
        plt.xlabel('Simulation Iteration')
        plt.ylabel('Reward Variance')
        plt.title('Reward Variance per Simulation')
        plt.legend()

        plt.tight_layout()
        if title is not None:
            plt.suptitle(title)
        if filename:
            plt.savefig(filename)
        plt.show()

# Example usage with CartPole-v1 environment
env = gym.make("CartPole-v1")
mcts = MCTS(env, num_simulations=10000, convergence_threshold=0.0005)
state = env.reset()
done = False

while not done:
    action = mcts.search(state)
    state, reward, done, _, _ = env.step(action)
    env.render()  # Render after each step
    time.sleep(0.05)  # Add a small delay to visualize the movement
    if done:
        print(f"Finished with reward: {reward}")

env.close()

# Plot performance metrics with moving average window size of 100
mcts.plot_performance(N=100, title="MCTS Performance Metrics", filename="mcts_performance.png")
