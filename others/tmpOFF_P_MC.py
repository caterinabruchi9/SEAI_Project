import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

def generate_episode(env, policy, epsilon=0.1, render=False):
    states, actions, rewards = [], [], []
    state = env.reset()  # Assumi che il reset restituisca solo lo stato iniziale
    if isinstance(state, tuple):  # Verifica se lo stato è una tupla e estrai solo il componente numerico necessario
        state = state[0] if isinstance(state[0], np.ndarray) else np.array(state[0])

    done = False
    while not done:
        if render:
            env.render()
        if torch.rand(1).item() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = policy(state_tensor)
            action = torch.argmax(action_probs).item()

        result = env.step(action)
        next_state = result[0] if isinstance(result, tuple) else result
        if isinstance(next_state, tuple):
            next_state = next_state[0] if isinstance(next_state[0], np.ndarray) else np.array(next_state[0])
        reward, done = result[1], result[2]

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return states, actions, rewards


def train_policy_gradient(env, policy, episodes, gamma=0.99, epsilon=0.1, render=False):
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    for episode in range(episodes):
        states, actions, rewards = generate_episode(env, policy, epsilon, render=(episode % 50 == 0))
        returns = compute_returns(rewards, gamma)

        update_policy(policy, optimizer, states, actions, returns)

        print(f"Episode {episode}: Total reward = {sum(rewards)}")
    env.close()

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std() + 1e-9)

def update_policy(policy, optimizer, states, actions, returns):
    states = torch.from_numpy(np.array(states, dtype=np.float32))
    actions = torch.from_numpy(np.array(actions, dtype=np.int64))

    action_probs = policy(states)
    target_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze().log()
    policy_loss = (-target_probs * returns).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

# Creazione dell'ambiente e addestramento
env = gym.make('CartPole-v1', render_mode='human')  # Assicurati che render_mode sia impostato qui
policy = PolicyNetwork()
train_policy_gradient(env, policy, 500)
