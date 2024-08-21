import numpy as np
import gym
import time

class NaiveMonteCarlo:
    def __init__(self, env, num_simulations=1000):
        self.env = env
        self.num_simulations = num_simulations

    def _simulate(self, action):
        env_copy = gym.make(self.env.spec.id)
        state = env_copy.reset()
        total_reward = 0
        done = False

        # Apply the initial action
        state, reward, done, _, _ = env_copy.step(action)
        total_reward += reward

        # Continue simulation until done
        while not done:
            action = np.random.choice(self.env.action_space.n)
            state, reward, done, _, _ = env_copy.step(action)
            total_reward += reward
        
        return total_reward


    def search(self):
        action_values = np.zeros(self.env.action_space.n)
        
        for action in range(self.env.action_space.n):
            total_reward = 0
            for _ in range(self.num_simulations):
                reward = self._simulate(action)
                total_reward += reward
            action_values[action] = total_reward / self.num_simulations
        
        return np.argmax(action_values)

env = gym.make("CartPole-v1", render_mode="human")
mc_search = NaiveMonteCarlo(env, num_simulations=100)

state = env.reset()
done = False

while not done:
    action = mc_search.search()
    state, reward, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.05)  # Add a small delay to visualize the movement
    if done:
        print(f"Finished with reward: {reward}")

env.close()
