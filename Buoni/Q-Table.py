import gym
import numpy as np
import random
import math
from abc import ABC
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        
        # Discretization parameters
        self.buckets = (1, 1, 6, 12)  # Number of buckets for each state variable
        self.state_size = np.prod(self.buckets)

        # Q-table initialized to zeros
        self.table = np.zeros(self.buckets + (self.action_size,))
        self.score = []

        # To store returns for each state-action pair
        self.returns_sum = {}
        self.returns_count = {}

        # Discretization bounds
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-math.radians(50), math.radians(50)]

    def discretize_state(self, state):
        discretized = []
        for i in range(len(state)):
            scaling = (state[i] - self.state_bounds[i][0]) / (self.state_bounds[i][1] - self.state_bounds[i][0])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

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
        discretized_state = self.discretize_state(state)
        return np.argmax(self.table[discretized_state])

    def learn(self):
        for e in tqdm(np.arange(self.episodes), desc="Learning"):
            state, _ = self.env.reset()
            discretized_state = self.discretize_state(state)
            episode_score = 0
            done = False
            episode = []

            while not done:
                if self.render_during_learning:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                discretized_next_state = self.discretize_state(next_state)

                episode_score += reward
                episode.append((discretized_state, action, reward))
                state = next_state
                discretized_state = discretized_next_state

                if done:
                    self.update_table(episode)
                    self.update_epsilon()
                    self.score.append(episode_score)

    def plot_learning(self, N, title=None, filename=""):
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


    def save_model(self, path):
        np.save(path, self.table)

    def load_model(self, path):
        self.table = np.load(path)

    def simulate(self):
        env = gym.make(self.env_name, render_mode="human")
        self.epsilon = -1
        done = False
        state, _ = env.reset()
        while not done:
            env.render()
            action = self.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

        env.close()

if __name__ == '__main__':
    chosen = "every-visit"
    agent = tabular_agent(env='CartPole-v1', epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01, episodes=5000, gamma=0.99, strategy=chosen, render_during_learning=True)
    agent.learn()
    agent.plot_learning(N=100, title="Learning Curve", filename=chosen + "_learning_curve.png")
    agent.simulate()
