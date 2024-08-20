import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class MonteCarloGPI:

    def __init__(self, env_name, episodes, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        
        self.Q = defaultdict(lambda: np.zeros(self.action_space))
        self.returns_sum = defaultdict(lambda: np.zeros(self.action_space))
        self.returns_count = defaultdict(lambda: np.zeros(self.action_space))
        self.policy = defaultdict(int)
        self.scores = []
    
    def discretize_state(self, state):
        bins = [np.linspace(-4.8, 4.8, 10), 
                np.linspace(-4, 4, 10), 
                np.linspace(-.418, .418, 10), 
                np.linspace(-4, 4, 10)]
        state_indices = tuple(np.digitize(s, bins[i]) for i, s in enumerate(state))
        return state_indices
    
    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
    
    def update_policy(self):
        for state in self.Q:
            self.policy[state] = np.argmax(self.Q[state])
    
    def generate_episode(self):
        episode = []
        state = self.discretize_state(self.env.reset()[0])
        done = False
        while not done:
            action = self.epsilon_greedy_policy(state)
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = self.discretize_state(next_state)
            episode.append((state, action, reward))
            state = next_state
        return episode
    
    def update_value_function(self, episode):
        G = 0
        visited_state_action_pairs = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited_state_action_pairs:
                self.returns_sum[state][action] += G
                self.returns_count[state][action] += 1.0
                self.Q[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
                visited_state_action_pairs.add((state, action))
    
    def learn(self):
        for episode in range(self.episodes):
            episode_data = self.generate_episode()
            self.update_value_function(episode_data)
            self.update_policy()
            self.scores.append(sum([reward for _, _, reward in episode_data]))
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if (episode + 1) % 100 == 0:
                print(f"Episode: {episode + 1}, Score: {self.scores[-1]}, Epsilon: {self.epsilon}")
    
    def simulate(self):
        state = self.discretize_state(self.env.reset()[0])
        done = False
        total_reward = 0
        while not done:
            self.env.render()
            action = self.policy[state]
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = self.discretize_state(next_state)
            total_reward += reward
            state = next_state
        print(f"Total Reward: {total_reward}")
        self.env.close()

    def plot_learning(self):
        plt.plot(self.scores)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.show()

if __name__ == '__main__':
    agent = MonteCarloGPI(env_name='CartPole-v1', episodes=5000)
    agent.learn()
    agent.plot_learning()
    agent.simulate()
