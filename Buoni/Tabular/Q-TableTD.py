import gymnasium as gym
from abc import ABC
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

class tabular_agent(ABC):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma):
        self.env_name = env
        self.env = gym.make(self.env_name)
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.gamma = gamma
        
        self.action_space = self.env.action_space
        self.action_size = self.env.action_space.n
        
        # For CliffWalking-v0, state space is discrete, so no binning needed
        self.state_size = self.env.observation_space.n
        
        # Definition of the Q-table
        self.table = np.zeros([self.state_size, self.action_size])
        self.score = []

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_table(self, state, action, reward, next_state, done):
        pass
    
    def select_action(self, state):  # epsilon-greedy action selection
        if np.random.random() <= self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.table[state])

    def learn(self):
        for e in tqdm(range(self.episodes), desc="Learning"):
            state, _ = self.env.reset()
            episode_score = 0
            done = False
            
            while not done:
                # Choose the next action
                action = self.select_action(state)
                
                # Do the chosen action
                next_state, reward, done, _, _ = self.env.step(action)
                
                episode_score += reward

                # Learn
                self.update_table(state, action, reward, next_state, done)

                # Update the current state
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
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

        env.close()


class Q_Learning(tabular_agent):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma):
        super().__init__(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma)

    def update_table(self, state, action, reward, next_state, done):
        if done:
            self.table[state][action] += reward - self.table[state][action]
        else:
            self.table[state][action] += reward + self.gamma * np.max(self.table[next_state]) - self.table[state][action]


class SARSA(tabular_agent):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma):
        super().__init__(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma)

    def update_table(self, state, action, reward, next_state, done):
        next_action = self.select_action(next_state)
        if done:
            self.table[state][action] += reward - self.table[state][action]
        else:
            self.table[state][action] += reward + self.gamma * self.table[next_state][next_action] - self.table[state][action]


class Expected_SARSA(tabular_agent):

    def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma):
        super().__init__(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma)

    def update_table(self, state, action, reward, next_state, done):
        if done:
            self.table[state][action] += reward - self.table[state][action]
        else:
            # Calculate the expected value for the next state
            expected_value = np.mean(self.table[next_state])
            self.table[state][action] += reward + self.gamma * expected_value - self.table[state][action]


if __name__ == '__main__':
    env = 'CliffWalking-v0'  # Environment updated for CliffWalking
    epsilon_start = 1.0
    epsilon_decay = 0.995  # Slower decay for more episodes
    epsilon_min = 0.01
    episodes = 1000  # Adjust as needed
    gamma = 0.99
    
    # Choose the agent class you want to use
    agent = Expected_SARSA(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma)
    
    agent.learn()
    agent.plot_learning(11, filename="_Expected_SARSA")
    
    # Save the trained model
    agent.save_model("models/cliffwalking_expected_sarsa.npy")
    
    agent.simulate()
