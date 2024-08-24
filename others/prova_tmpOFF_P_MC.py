import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class PolicyAgent:
    def __init__(self, env_name, episodes, gamma=0.99, epsilon_start=0.1, epsilon_decay=0.99, epsilon_min=0.01):
        self.env = gym.make(env_name)
        self.policy_net = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        #ADAM: basato sul gradiente, minimizzare la loss function
        #momento: media mobile del grediente, serve per non considerare piccole variazioni
        #POI: calcola il quadrato di ciascun componenete del gradiente e ne fa una media mobile nel tempo
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)

        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rewards_history = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploration
        else:
            #check sul formato di state altrimenti fa casino
            state = state if isinstance(state, np.ndarray) else state[0]
            with torch.no_grad():
                return torch.argmax(
                    self.policy_net(torch.from_numpy(state).float().unsqueeze(0))).item()  # Exploitation

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        return (returns - returns.mean()) / (returns.std() + 1e-9)

    def update_policy(self, states, actions, returns):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        action_probs = self.policy_net(states)
        log_probs = action_probs.gather(1, actions.unsqueeze(1)).log()

        loss = -(log_probs.squeeze() * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for episode in tqdm(range(self.episodes), desc="Training Episodes"):
            states, actions, rewards = self.run_episode()
            returns = self.compute_returns(rewards)
            self.update_policy(states, actions, returns)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.rewards_history.append(sum(rewards))
            if episode % 10 == 0:
                print(f'Episode {episode}: Total Reward = {sum(rewards)}')

    def run_episode(self):
        states, actions, rewards = [], [], []
        state = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state)
            result = self.env.step(action)  # Handle a variable number of return values flexibly
            next_state, reward, done, *_ = result
            states.append(state if isinstance(state, np.ndarray) else state[0])
            actions.append(action)
            rewards.append(reward)
            state = next_state
        return states, actions, rewards

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

if __name__ == '__main__':
    agent = PolicyAgent('CartPole-v1', episodes=500)
    agent.train()
    agent.plot_rewards()
