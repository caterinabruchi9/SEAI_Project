import gymnasium as gym

from abc import ABC
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep

import numpy as np
import math, random
import os

class tabular_agent(ABC):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins=10):
        self.env_name = env
        self.env = gym.make(self.env_name)
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.gamma = gamma
        
        self.action_space = self.env.action_space
        self.action_size = self.env.action_space.n
        
        # Discretization of the observation space
        self.n_bins = n_bins
        self.state_bins = [
            np.linspace(-4.8, 4.8, self.n_bins),  # Cart position
            np.linspace(-4, 4, self.n_bins),     # Cart velocity
            np.linspace(-0.418, 0.418, self.n_bins),  # Pole angle
            np.linspace(-4, 4, self.n_bins)      # Pole angular velocity
        ]

        # Definition of the Q-table
        self.state_size = [self.n_bins] * len(self.state_bins)
        self.table = np.zeros(self.state_size + [self.action_size])
        self.score = []

    def discretize_state(self, state):
        """Discretizes the continuous state into discrete bins."""
        state_indices = []
        for i in range(len(state)):
            state_indices.append(np.digitize(state[i], self.state_bins[i]) - 1)
        return tuple(state_indices)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_table(self, state, action, next_state, reward, done):
        pass
    
    def select_action(self, state): # epsilon-greedy action selection
        if random.random() <= self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.table[state])

    def learn(self):
        for e in tqdm(np.arange(self.episodes), desc="Learning"):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            episode_score = 0
            done = False
            
            while not done:
                # choose the next action
                action = self.select_action(state)
                
                # do the chosen action
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)

                episode_score += reward

                # learn
                self.update_table(state, action, reward, next_state, done)

                # update the current state
                state = next_state

            self.update_epsilon()
            self.score.append(episode_score)

    def plot_learning(self, N, title=None, filename=""):
        plt.figure()
        
        # Compute the moving average of the score
        mean_score = np.convolve(np.array(self.score), np.ones(N)/N, mode='valid')
        
        # Adjust x values for the moving average plot
        x_values = np.arange(len(mean_score)) + (N // 2)

        # Plot the original scores
        plt.plot(self.score, label='Score')
        
        # Plot the moving average
        plt.plot(x_values, mean_score, label='Moving Average', color='orange')
        
        if title is not None:
            plt.title(title)
        plt.legend()
        plt.savefig(self.env_name + filename)
        plt.show()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.table)
        print(f"Q-table saved to {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.table = np.load(path)
            print(f"Q-table loaded from {path}")
        else:
            print(f"File {path} does not exist.")

    def simulate(self):
        env = gym.make(self.env_name, render_mode="human")
        self.epsilon = -1
        done = False
        state, _ = env.reset()
        state = self.discretize_state(state)
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = self.discretize_state(next_state)

            state = next_state

        env.close()


class Q_Learning(tabular_agent):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins=10):
        super().__init__(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins)

    def update_table(self, state, action, reward, next_state, done):
        if done:
            self.table[state][action] += reward - self.table[state][action]
        else:
            self.table[state][action] += reward + self.gamma * np.max(self.table[next_state]) - self.table[state][action]


class SARSA(tabular_agent):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins=10):
        super().__init__(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins)

    def update_table(self, state, action, reward, next_state, done):
        next_action = self.select_action(next_state)
        if done:
            self.table[state][action] += reward - self.table[state][action]
        else:
            self.table[state][action] += reward + self.gamma * self.table[next_state][next_action] - self.table[state][action]
            
class Expected_SARSA(tabular_agent):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins=10):
        super().__init__(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins)

    def update_table(self, state, action, reward, next_state, done):
        if done:
            self.table[state][action] += reward - self.table[state][action]
        else:
            expected_value = np.mean(self.table[next_state])
            self.table[state][action] += reward + self.gamma * expected_value - self.table[state][action]


if __name__ == '__main__':
    env = 'CartPole-v1'  # Updated environment
    epsilon_start = 1.0
    epsilon_decay = 0.995  # Slower decay for more episodes
    epsilon_min = 0.01
    episodes = 1000  # Increased number of episodes for CartPole
    gamma = 0.99
    
    agent = Expected_SARSA(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, n_bins=10)
    
    
    
    agent.learn()
    agent.plot_learning(11, filename="_Expected_SARSA")
    
    # Save the trained model
    agent.save_model("models/cartpole_expected_sarsa.npy")
    
    agent.simulate()
