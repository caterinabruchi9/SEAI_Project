import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time

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

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action, next_state):
        child_node = MCTSNode(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTS:
    def __init__(self, env, num_simulations=1000, convergence_threshold=0.01):
        self.env = env
        self.num_simulations = num_simulations
        self.convergence_threshold = convergence_threshold
        self.root = None

        # Tracking performance metrics
        self.avg_rewards = []
        self.best_values = []
        self.best_visits = []

    def _simulate(self, env_copy):
        done = False
        total_reward = 0
        while not done:
            action = np.random.choice(self.env.action_space.n)
            _, reward, done, _, _ = env_copy.step(action)
            total_reward += reward
        return total_reward

    def search(self, initial_state):
        self.root = MCTSNode(state=initial_state)
        previous_best_value = float('inf')  # Cambiato da '-inf' a 'inf'

        for i in range(self.num_simulations):
            node = self.root
            env_copy = gym.make(self.env.spec.id)
            env_copy.reset()

            # Selection
            while node.is_fully_expanded(self.env.action_space.n) and node.children:
                node = node.best_child()
                action = node.action
                _, _, done, _, _ = env_copy.step(action)
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
            if not done:
                reward = self._simulate(env_copy)
            else:
                reward = 0  # or some other logic depending on your goal

            # Backpropagation
            self._backpropagate(node, reward)

            # Record metrics
            best_child = self.root.best_child(c_param=0.0)
            best_value = best_child.value / best_child.visits if best_child.visits > 0 else float('-inf')
            avg_reward = reward
            self.avg_rewards.append(avg_reward)
            self.best_values.append(best_value)
            self.best_visits.append(best_child.visits if best_child.visits > 0 else 0)

            # Debug information
            print(f"Iteration {i}: Best Value = {best_value}, Previous Best Value = {previous_best_value}, Avg Reward = {avg_reward}")

            # Check for convergence
            if abs(best_value - previous_best_value) < self.convergence_threshold:
                print(f"Convergence reached with value: {best_value}")
                break
            previous_best_value = best_value

        return self.root.best_child(c_param=0.0).action

    def _select_untried_action(self, node):
        tried_actions = [child.action for child in node.children]
        possible_actions = set(range(self.env.action_space.n)) - set(tried_actions)
        return np.random.choice(list(possible_actions))

    def _backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def plot_performance(self):
        plt.figure(figsize=(12, 6))

        # Plot average rewards
        plt.subplot(1, 3, 1)
        plt.plot(self.avg_rewards, label='Average Reward')
        plt.xlabel('Simulation Iteration')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per Simulation')
        plt.legend()

        # Plot best values
        plt.subplot(1, 3, 2)
        plt.plot(self.best_values, label='Best Value', color='orange')
        plt.xlabel('Simulation Iteration')
        plt.ylabel('Best Value')
        plt.title('Best Value Estimate')
        plt.legend()

        # Plot visit counts
        plt.subplot(1, 3, 3)
        plt.plot(self.best_visits, label='Best Visits', color='green')
        plt.xlabel('Simulation Iteration')
        plt.ylabel('Visit Count')
        plt.title('Visit Counts of Best Actions')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Example usage with CartPole-v1 environment
env = gym.make("CartPole-v1", render_mode="human")  # Set render_mode to "human"
mcts = MCTS(env, num_simulations=10000, convergence_threshold=0.001)
state, _ = env.reset()
done = False

while not done:
    action = mcts.search(state)
    state, reward, done, _ ,_ = env.step(action)
    env.render()  # Render after each step
    time.sleep(0.05)  # Add a small delay to visualize the movement
    if done:
        print(f"Finished with reward: {reward}")

env.close()

# Plot performance metrics
mcts.plot_performance()
