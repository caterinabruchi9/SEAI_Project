import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class REINFORCE_UTB:
    def __init__(self, env_name, gamma=0.99, lr=1e-2, episodes=1000, K=5):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.lr = lr
        self.episodes = episodes
        self.K = K
        
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        self.policy_net = PolicyNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.episode_rewards = []
        self.baseline_rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_net(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update_policy(self, log_probs, returns):
        returns = torch.tensor(returns)
        
        # Calculate baseline as the mean of the top-K returns
        sorted_returns, _ = torch.sort(returns, descending=True)
        baseline = sorted_returns[:self.K].mean()
        
        # Subtract the baseline from the returns to get advantages
        advantages = returns - baseline
        
        loss = -torch.sum(torch.stack(log_probs) * advantages)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self):
        for episode in tqdm(range(self.episodes)):
            state, _ = self.env.reset()
            state = torch.eye(self.state_size)[state]
            
            log_probs = []
            rewards = []
            done = False

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = torch.eye(self.state_size)[next_state]
                
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state
                done = terminated or truncated

            returns = self.compute_returns(rewards)
            self.episode_rewards.append(sum(rewards))
            self.update_policy(log_probs, returns)
            
            # Store the baseline reward
            self.baseline_rewards.append(np.mean(sorted(returns)[-self.K:]))
        
        self.env.close()

    def plot_rewards(self, smoothing_window=10):
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards, label="Episode Rewards")
        smoothed_rewards = np.convolve(self.episode_rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
        plt.plot(smoothed_rewards, label="Smoothed Rewards", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Performance with REINFORCE + UTB")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    agent = REINFORCE_UTB(env_name, gamma=0.99, lr=1e-2, episodes=1000, K=5)
    agent.train()
    agent.plot_rewards()
