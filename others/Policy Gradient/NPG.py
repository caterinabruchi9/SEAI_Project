import numpy as np
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

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

def compute_fisher_information_matrix(states, actions, policy_net):
    fisher_matrix = np.zeros((policy_net.fc3.out_features, policy_net.fc3.out_features))
    for state, action in zip(states, actions):
        state_tensor = state_to_tensor(state)
        probs = policy_net(state_tensor).detach().numpy().flatten()
        pi = torch.FloatTensor(probs)
        advantage = pi[action] - pi.mean()
        fisher_matrix += np.outer(advantage, advantage)
    return fisher_matrix

def compute_policy_gradient(states, actions, returns, policy_net):
    log_probs = []
    for state, action in zip(states, actions):
        state_tensor = state_to_tensor(state)
        probs = policy_net(state_tensor)
        log_prob = torch.log(probs[0, action])
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs)
    returns_tensor = torch.FloatTensor(returns)
    policy_gradient = -torch.mean(log_probs * returns_tensor)
    return policy_gradient.item()

def natural_policy_gradient(policy_net, states, actions, returns):
    fisher_matrix = compute_fisher_information_matrix(states, actions, policy_net)
    fisher_matrix_inv = np.linalg.pinv(fisher_matrix + 1e-8 * np.eye(fisher_matrix.shape[0]))
    
    policy_grad = compute_policy_gradient(states, actions, returns, policy_net)
    
    natural_grad = fisher_matrix_inv @ policy_grad
    return natural_grad

def update_policy(policy_net, natural_grad, learning_rate=0.01):
    with torch.no_grad():
        for param in policy_net.parameters():
            param += learning_rate * torch.FloatTensor(natural_grad)

def state_to_tensor(state):
    if isinstance(state, tuple):
        state = state[0]  # Estrarre solo il primo elemento della tupla
    state_tensor = torch.FloatTensor(state)
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)
    return state_tensor

def run_mountain_car(env_name='MountainCar-v0', num_episodes=1000, gamma=0.99, lr=0.01, window_size=10):
    env = gym.make(env_name, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNetwork(state_dim, action_dim)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        states, actions, rewards = [], [], []
        
        while not done:
            state_tensor = state_to_tensor(state)
            action_probs = policy_net(state_tensor).detach().numpy().flatten()
            action = np.random.choice(action_dim, p=action_probs)
            
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        # Compute natural gradient
        natural_grad = natural_policy_gradient(policy_net, states, actions, returns)
        
        # Update policy
        update_policy(policy_net, natural_grad, learning_rate=lr)
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # Save the policy network weights
    torch.save(policy_net.state_dict(), 'policy_network_weights.pth')
    print("Policy network weights saved to 'policy_network_weights.pth'")

    # Compute rewards variance over a rolling window
    rewards_array = np.array(episode_rewards)
    rolling_mean = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    rolling_var = np.array([np.var(rewards_array[max(0, i-window_size+1):i+1]) for i in range(len(rewards_array))])
    
    # Plot rewards and variance
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_array, label='Episode Rewards', color='blue')
    plt.plot(rolling_mean, label=f'Rolling Mean (window={window_size})', color='orange')
    plt.plot(rolling_var, label='Rolling Variance', color='red', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('MountainCar-v0 Training Progress')
    plt.legend()
    plt.show()

    env.close()

# Esegui l'algoritmo su MountainCar-v0
run_mountain_car()
