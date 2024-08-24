import sys
import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os  # Import os module for path operations

def random_policy(nA):
    return lambda observation: np.ones(nA, dtype=float) / nA

def epsilon_greedy_policy(Q, epsilon=0.1):
    def policy_fn(state):
        if isinstance(state, tuple):
            state = state[0]
        nA = len(Q[state])
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

class MC_offpolicy:
    def __init__(self, env, num_episodes, discount_factor=0.99, epsilon=0.05):
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        #Contatore coppie stato-azione per mediare i ritorni incrementali
        self.C = defaultdict(lambda: np.zeros(env.action_space.n))
        #Policy esplorativa
        self.behavior_policy = random_policy(env.action_space.n)
        #Policy da ottimizzare ed usare
        self.target_policy = epsilon_greedy_policy(self.Q, epsilon=self.epsilon)

    def update(self, episode, episode_num):
        #QUESTO VIENE FATTO PER OGNI EPISODIO
        print(f"UPDATE EPISODE: {episode_num}\n")
        #UPDATE RULE:
        # Q(s_t, a_t) = Q(s_t,a_t) + v(G_t^pi - Q(s_t,a_t))
        # con v = (W)/(C(s_t,a_t))

        #Per ogni coppia stato-azione di un epsidio aggiorniamo G
        # Viene calcolato al contrario partendo dalla fine andando all'inizio
        G = 0.0
        #IMPORTANCE SAMPLING:
        #Andiamo a pesare la differenza tra policy esplorativa e policy target
        W = 1.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            if isinstance(state, tuple):
                state = state[0]
            G = self.discount_factor * G + reward
            self.C[state][action] += W
            self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
            if action != np.argmax(self.target_policy(state)):
                break
            W /= self.behavior_policy(state)[action]

    def train(self, verbose=True):
        #4) Train
        print(f"4) Train\n")
        #Ogni volta raccoglie s,a,r e
        # aggiorna Q

        #Vettore delle ricompense
        rewards = []
        for trial in range(1, self.num_episodes + 1):
            print(f"5) RUNNING EPISODE NUMBER: {trial}\n")
            state = self.env.reset()
            """
            if isinstance(state, tuple):
                print(f"State is instance of tuple: {state}")
                state = state[0]
            """
            state = state[0]
            done = False
            episode = []
            while not done:
                """if render:
                    self.env.render()
                """
                # print(f"PROB VAL: {self.behavior_policy(state)}") è sempre 0.25
                action_probs = self.behavior_policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, other_bool, info = self.env.step(action)
                print(f"\nNEXT_STATE: {state}\nREWARD: {reward}\nDONE: {done}\nOTHER: {other_bool}\nINFO: {info}")
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                episode.append((state, action, reward))
                state = next_state
            self.update(episode, trial)

            total_reward = sum(x[2] for x in episode)
            rewards.append(total_reward)
        # print(f"VERBOSE: {verbose}")
        if verbose:
            # print(f"REWARDS: {rewards}")
            plt.plot(rewards)
            plt.title('Rewards per episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plot_path = os.path.join(os.getcwd(), 'reward_plot.png')
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            plt.show()
        return np.mean(rewards)

    def render_policy(self):
        new_env = gym.make("FrozenLake-v1", render_mode="human")
        # self.epsilon =-1
        state = new_env.reset()
        if isinstance(state, dict):
            state = state['observation']
        new_env.render()
        done = False
        while not done:
            action = np.argmax(self.target_policy(state))
            state, reward, done, other_bool, info = new_env.step(action)
            if isinstance(state, dict):
                state = state['observation']
            new_env.render()
            time.sleep(1)
        print("Episode finished.")

# Setup and run
print(f"1) Application starts\n")
# 2) Environment setting
env = gym.make("FrozenLake-v1", is_slippery=False)
rl = MC_offpolicy(env, num_episodes=10000, epsilon=0.05)
#3) Run
average_reward = rl.train(verbose=True)
print(f"Average Reward after 500 episodes: {average_reward}")
rl.render_policy()