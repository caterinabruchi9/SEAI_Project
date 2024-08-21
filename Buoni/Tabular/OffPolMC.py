import sys
import numpy as np
import gym
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import time

def random_policy(nA):
    def policy_fn(observation):
        return np.ones(nA, dtype=float) / nA
    return policy_fn

def epsilon_greedy_policy(Q):
    def policy_fn(state):
        nA = len(Q[str(state)])
        A = np.ones(nA, dtype=float) * 0.05 / (nA - 1)
        best_action = np.argmax(Q[str(state)])
        A[best_action] = 0.85
        # Ensure probabilities sum to 1
        A /= A.sum()
        return A
    return policy_fn

class MC_offpolicy:
    def __init__(self, env, num_episodes, discount_factor=0.99):
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.C = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.behavior_policy = random_policy(self.env.action_space.n)
        self.target_policy = epsilon_greedy_policy(self.Q)

    def update(self, episode):
        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            
            G = self.discount_factor * G + reward
            self.C[str(state)][action] += 1
            old_value = self.Q[str(state)][action]
            self.Q[str(state)][action] += (W / (self.C[str(state)][action] + 1e-60)) * (G - old_value)
            if action == np.argmax(self.target_policy(state)):
                W *= 0.85 / 0.05
            else:
                W *= 0.05 / 0.05
            # Debug information
            print(f"Update: State: {state}, Action: {action}, Reward: {reward}, G: {G}, W: {W}, Old Value: {old_value}, New Value: {self.Q[str(state)][action]}")

    

    def simulate(self, render=False, train=True, verbose=True):
        b_rewards = []
        t_rewards = []
        episode_length = []
        mean_rewards = []
        max_totalReward = 0

        if train:
            for trial in range(1, self.num_episodes + 1):
                if trial % 1000 == 0:
                    print(f"\rEpisode {trial}/{self.num_episodes}.", end="")
                    sys.stdout.flush()
                
                episode = []
                state = self.env.reset()
                done = False
                step = 0
                while not done:
                    probs = self.behavior_policy(state)
                    if not np.isclose(np.sum(probs), 1.0):
                        raise ValueError(f"Behavior policy probabilities do not sum to 1: {probs}")
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    next_state, reward, done, _ , _ = self.env.step(action)
                    episode.append((state, action, reward))
                    state = next_state
                    step += 1
                    # Debug information
                    if verbose and step % 100 == 0:
                        print(f"Episode {trial}, Step {step}: State: {state}, Action: {action}, Reward: {reward}")

                self.update(episode)
                total_reward = sum([reward for _, _, reward in episode])
                b_rewards.append(total_reward)

                state = self.env.reset()
                total_reward = 0
                iteration = 0
                done = False
                while not done:
                    probs = self.target_policy(state)
                    if not np.isclose(np.sum(probs), 1.0):
                        raise ValueError(f"Target policy probabilities do not sum to 1: {probs}")
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    next_state, reward, done, _ , _ = self.env.step(action)

                    if render:
                        self.env.render()  # Call render without checking return value
                        time.sleep(0.05)  # Add a slight delay to make rendering visible

                    total_reward += reward
                    state = next_state
                    iteration += 1
                    # Debug information
                    if verbose and iteration % 100 == 0:
                        print(f"Trial {trial}, Target Policy Step {iteration}: State: {state}, Action: {action}, Reward: {reward}")

                    if done:
                        break
                
                episode_length.append(iteration)
                t_rewards.append(total_reward)
                mean_rewards.append(np.mean(t_rewards[-100:]))  # Average reward over last 100 episodes
                
                if verbose and trial % 20 == 0:
                    print(f'\n---- Trial {trial} ----')
                    print(f'Mean(last 10 total rewards): {np.mean(t_rewards[-10:])}')
                
                mean_totalReward = np.mean(t_rewards[-10:])
                if mean_totalReward > max_totalReward and train:
                   
                    max_totalReward = mean_totalReward
                    print(f'The weights are saved with total rewards: {mean_totalReward}')
        
        print(f'Trial {trial} Total Reward: {t_rewards[-1]}')

        # Plot performance analysis
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(t_rewards, label='Total Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(mean_rewards, label='Mean Reward (last 100 episodes)', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward (last 100 episodes)')
        plt.legend()

        plt.tight_layout()
        
        plt.show()

        return self.Q, self.target_policy

 
# Example usage
env = gym.make("CartPole-v1", render_mode="human") 
rl = MC_offpolicy(env, num_episodes=1000)
Q, target_policy = rl.simulate(render=True, train=True, verbose=True)
