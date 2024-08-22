import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from abc import ABC
from tqdm import tqdm

class tabular_agent(ABC):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, strategy="first-visit", render_during_learning=False):
        self.env_name = env
        self.env = gym.make(self.env_name)
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.gamma = gamma
        self.strategy = strategy
        self.render_during_learning = render_during_learning

        self.action_space = self.env.action_space
        self.action_size = self.env.action_space.n

        # Q-table initialized to zeros (16x4 for CliffWalking-v0, as it has 16 states and 4 actions)
        self.table = np.zeros((self.env.observation_space.n, self.action_size))
        self.score = []

        # To store returns for each state-action pair
        self.returns_sum = {}
        self.returns_count = {}

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_table(self, episode):
        G = 0  # Initialize the return
        visited_state_action_pairs = set()

        # Iterate backwards through the episode
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward

            # Every-Visit Monte Carlo: Always update the Q-value
            if self.strategy == "every-visit" or (self.strategy == "first-visit" and (state, action) not in visited_state_action_pairs):
                if (state, action) not in self.returns_sum:
                    self.returns_sum[(state, action)] = 0.0
                    self.returns_count[(state, action)] = 0.0

                self.returns_sum[(state, action)] += G
                self.returns_count[(state, action)] += 1.0

                self.table[state][action] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

                visited_state_action_pairs.add((state, action))

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.table[state])

    def learn(self):
        for e in tqdm(np.arange(self.episodes), desc="Learning"):
            state, _ = self.env.reset()
            episode_score = 0
            done = False
            episode = []

            while not done:
                if self.render_during_learning:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                episode_score += reward
                episode.append((state, action, reward))
                state = next_state

                if done:
                    self.update_table(episode)
                    self.update_epsilon()
                    self.score.append(episode_score)

    def plot_learning(self, N, title=None, filename="Q-Table/"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.score, label="Score per Episode")
        
        # Calculate the moving average using convolution
        mean_score = np.convolve(np.array(self.score), np.ones(N) / N, mode='valid')
        
        # The x-axis for the moving average should match the length of mean_score
        x_range = np.arange(len(mean_score))
        
        plt.plot(x_range, mean_score, label=f"Moving Average (N={N})", color="orange")
        
        if title is not None:
            plt.title(title)
        
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(self.env_name + filename)
        plt.show()

    def plot_policy(self, filename="policy.png"):
        # Define the grid size for CliffWalking-v0 (4 rows x 12 columns)
        grid_height = 4
        grid_width = 12
        
        # Create a grid for the policy
        policy_grid = np.zeros((grid_height, grid_width), dtype=int)

        # Populate the policy grid with the action that has the highest Q-value
        for state in range(grid_height * grid_width):
            action = np.argmax(self.table[state])
            policy_grid[state // grid_width, state % grid_width] = action
        
        # Create a plot for the policy
        plt.figure(figsize=(10, 5))
        plt.imshow(policy_grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Action')
        plt.title('Learned Policy')
        plt.xlabel('Column')
        plt.ylabel('Row')

        # Map actions to colors
        action_labels = ['Left', 'Down', 'Right', 'Up']
        plt.xticks(ticks=np.arange(grid_width), labels=np.arange(grid_width))
        plt.yticks(ticks=np.arange(grid_height), labels=np.arange(grid_height))

        # Overlay action labels
        for i in range(grid_height):
            for j in range(grid_width):
                plt.text(j, i, action_labels[policy_grid[i, j]], ha='center', va='center', color='white')

        plt.savefig(filename)
        plt.show()

    def save_model(self, path):
        np.save(path, self.table)

    def load_model(self, path):
        self.table = np.load(path)

    def simulate(self):
        env = gym.make(self.env_name, render_mode="human")  # Use 'human' for interactive rendering
        self.epsilon = -1  # Disable exploration for policy demonstration
        state, _ = env.reset()
        done = False
        
        while not done:
            action = np.argmax(self.table[state])  # Select the best action
            next_state, reward, done, _, _ = env.step(action)
            
            state = next_state

            # Stop rendering if the terminal state (goal) is reached
            if done:
                env.render()  # Render the final state

       


    def run_experiment(self, plot_N=100, plot_title="Learning Curve", plot_filename="_learning_curve.png", model_path=None, policy_filename="policy.png"):
        # Run learning
        self.learn()
        
        # Plot the learning curve
        self.plot_learning(N=plot_N, title=plot_title, filename=plot_filename)
        
        # Plot the learned policy
        self.plot_policy(filename=policy_filename)
        
        # Save the model if path is provided
        if model_path:
            self.save_model(model_path)
        
        # Simulate and render the final solution
        self.simulate()

if __name__ == '__main__':
    chosen = "first-visit"
    agent = tabular_agent(env='CliffWalking-v0', epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01, episodes=5000, gamma=0.99, strategy=chosen, render_during_learning=False)
    agent.run_experiment(plot_N=100, plot_title="Learning Curve", plot_filename=chosen + "_learning_curve.png", model_path=chosen + "_q_table.npy", policy_filename=chosen + "_policy.png")
