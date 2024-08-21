import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Definizione della rete neurale per la politica
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def monte_carlo_policy_gradient(env_name='MountainCar-v0', num_episodes=1000, gamma=0.99, lr=0.01, render=False):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        state = env.reset()

        # Handle case where reset() returns a tuple
        if isinstance(state, tuple):
            state = state[0]

        done = False
        states, actions, rewards = [], [], []
        
        while not done:
            if render:
                env.render()  # Add rendering to visualize the environment

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action = np.random.choice(action_dim, p=action_probs.detach().numpy().flatten())
            
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state

            # Handle case where step() returns a tuple
            if isinstance(state, tuple):
                state = state[0]
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (optional but often helpful)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient
        policy_gradients = []
        for t in range(len(states)):
            state_tensor = torch.FloatTensor(states[t]).unsqueeze(0)
            action_prob = policy_net(state_tensor)[0, actions[t]]
            log_prob = torch.log(action_prob)
            policy_gradients.append(log_prob * returns[t])
        
        # Sum gradients and take optimization step
        loss = -torch.stack(policy_gradients).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 10 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

# Esegui l'algoritmo su MountainCar-v0 con rendering abilitato
monte_carlo_policy_gradient(render=True)
