import sys
import numpy as np
import gym
from collections import defaultdict
import pickle

def random_policy(nA):
    def policy_fn(observation):
        return np.ones(nA, dtype=float) / nA
    return policy_fn

def epsilon_greedy_policy(Q):
    def policy_fn(state):
        nA = len(Q[state])
        A = np.ones(nA, dtype=float) * 0.05 / (nA - 1)
        best_action = np.argmax(Q[state])
        A[best_action] = 0.85
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
            state = tuple(state)
            G = self.discount_factor * G + reward
            self.C[state][action] += W
            self.Q[state][action] += (W / (self.C[state][action] + 1e-60)) * (G - self.Q[state][action])
            if action == np.argmax(self.target_policy(state)):
                W *= 0.85 / 0.05
            else:
                W *= 0.05 / 0.05

    def simulate(self, render=False, train=True, verbose=True):
        b_rewards = []
        t_rewards = []
        max_totalReward = 0
        episode_length = []
        if train:
            for trial in range(1, self.num_episodes + 1):
                if trial % 1000 == 0:
                    print(f"\rEpisode {trial}/{self.num_episodes}.", end="")
                    sys.stdout.flush()
                
                episode = []
                state = self.env.reset()
                while True:
                    probs = self.behavior_policy(state)
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    next_state, reward, done = self.env.step(action)
                    episode.append((state, action, reward))
                    if done:
                        break
                    state = next_state
                
                self.update(episode)
                total_reward = sum([reward for _, _, reward in episode])
                b_rewards.append(total_reward)

                state = self.env.reset()
                total_reward = 0
                iteration = 0
                while True:
                    probs = self.target_policy(tuple(state))
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    next_state, reward, done, _ = self.env.step(action)

                    if render:
                        still_open = self.env.render()
                        if not still_open: break

                    total_reward += reward
                    state = next_state
                    iteration += 1
                    if done:
                        break
                
                episode_length.append(iteration)
                t_rewards.append(total_reward)
                if verbose and trial % 20 == 0:
                    print(f'\n---- Trial {trial} ----')
                    print(f'Mean(last 10 total rewards): {np.mean(t_rewards[-10:])}')
                mean_totalReward = np.mean(t_rewards[-10:])
                if mean_totalReward > max_totalReward and train:
                    self.saveF(self.Q, f'./weights/linear_MC_{trial}_{mean_totalReward}.pkl')
                    max_totalReward = mean_totalReward
                    print(f'The weights are saved with total rewards: {mean_totalReward}')
        
        print(f'Trial {trial} Total Reward: {t_rewards[-1]}')
        np.save(f"./rewards/t_off_MC_{self.num_episodes}.npy", t_rewards)
        np.save(f"./rewards/b_off_MC_{self.num_episodes}.npy", b_rewards)
        np.save("./episode_length.npy", episode_length)

        return self.Q, self.target_policy

    def saveF(self, obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def loadF(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

env = gym.make("MountainCar-v0")
rl = MC_offpolicy(env, num_episodes=10000)
Q, target_policy = rl.simulate(render=False, train=True, verbose=True)
