import numpy as np
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from scipy.optimize import minimize

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

def conjugate_gradient(Av_func, b, nsteps, residual_tol=1e-10):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = r.dot(r)

    for i in range(nsteps):
        Ap = Av_func(p)
        alpha = rdotr / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def fisher_vector_product(policy_net, states, actions, p):
    p = torch.FloatTensor(p)
    kl = 0
    for state, action in zip(states, actions):
        state_tensor = state_to_tensor(state)
        probs = policy_net(state_tensor)
        dist = Categorical(probs)
        kl += torch.sum(dist.probs * torch.log(dist.probs / dist.probs.detach()))

    kl_grad = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
    kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])
    kl_grad_p = (kl_grad * p).sum()

    kl_hessian = torch.autograd.grad(kl_grad_p, policy_net.parameters())
    kl_hessian = torch.cat([grad.contiguous().view(-1) for grad in kl_hessian])

    return kl_hessian + 1e-5 * p

def compute_policy_loss(policy_net, states, actions, returns):
    log_probs = []
    for state, action in zip(states, actions):
        state_tensor = state_to_tensor(state)
        probs = policy_net(state_tensor)
        log_prob = torch.log(probs[0, action])
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs)
    returns_tensor = torch.FloatTensor(returns)
    policy_loss = -torch.mean(log_probs * returns_tensor)
    return policy_loss

def trpo_step(policy_net, states, actions, returns, max_kl=1e-2, cg_iters=10, damping=1e-2):
    # Compute policy loss gradient
    loss = compute_policy_loss(policy_net, states, actions, returns)
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1).detach().numpy() for grad in grads])

    # Compute step direction using conjugate gradient
    def fisher_vector_product_func(p):
        return fisher_vector_product(policy_net, states, actions, p)

    step_dir = conjugate_gradient(fisher_vector_product_func, -loss_grad, nsteps=cg_iters)

    # Compute the scale of the step size
    shs = 0.5 * step_dir.dot(fisher_vector_product_func(step_dir))
    lagrange_multiplier = np.sqrt(shs / max_kl)
    fullstep = step_dir / lagrange_multiplier

    # Line search
    def line_search_fn(step):
        new_params = torch.cat([param.view(-1) for param in policy_net.parameters()]) + step
        set_params(policy_net, new_params)
        new_loss = compute_policy_loss(policy_net, states, actions, returns).item()
        return new_loss

    success, step = line_search(line_search_fn, fullstep, max_backtracks=10)

    # Update policy network parameters
    if success:
        new_params = torch.cat([param.view(-1) for param in policy_net.parameters()]) + step
        set_params(policy_net, new_params)

def line_search(f, x, max_backtracks=10, accept_ratio=0.1):
    fval = f(x)
    for stepfrac in .5**np.arange(max_backtracks):
        xnew = x * stepfrac
        newfval = f(xnew)
        if newfval < fval + accept_ratio * stepfrac * (x.T @ x):
            return True, xnew
    return False, x

def set_params(policy_net, new_params):
    start = 0
    for param in policy_net.parameters():
        end = start + param.numel()
        param.data.copy_(new_params[start:end].view(param.size()))
        start = end

def state_to_tensor(state):
    if isinstance(state, tuple):
        state = state[0]  # Estrarre solo il primo elemento della tupla
    state_tensor = torch.FloatTensor(state)
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)
    return state_tensor

def run_mountain_car_trpo(env_name='MountainCar-v0', num_episodes=1000, gamma=0.99, window_size=10):
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
        returns = compute_returns(rewards, gamma)
        
        # Perform TRPO step
        trpo_step(policy_net, states, actions, returns)
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # Save the policy network weights
    torch.save(policy_net.state_dict(), 'trpo_policy_network_weights.pth')
    print("Policy network weights saved to 'trpo_policy_network_weights.pth'")

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
    plt.title('MountainCar-v0 Training Progress (TRPO)')
    plt.legend()
    plt.show()

    env.close()

# Esegui l'algoritmo TRPO su MountainCar-v0
run_mountain_car_trpo()
