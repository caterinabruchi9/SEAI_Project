import sys
import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
import time

"""
def state_based_policy(nA):
    def policy_fn(observation):
        # Supponiamo che 'observation' possa avere un attributo che ci interessa
        # Per esempio, supponiamo che 'observation' abbia un valore numerico e che
        # vogliamo aumentare la probabilità di un'azione se questo valore è alto.
        probabilities = np.ones(nA, dtype=float) / nA  # Distribuzione di base uniforme
        if observation > threshold:  # Valuta lo stato corrente
            best_action = get_best_action_based_on_state(observation)  # Decidi la migliore azione basata su 'observation'
            probabilities *= 0.5  # Riduci tutte le probabilità
            probabilities[best_action] += 0.5  # Aumenta la probabilità della 'migliore' azione
        return probabilities

    return policy_fn

# Funzione fittizia per ottenere la migliore azione basata su uno stato
def get_best_action_based_on_state(observation):
    # Questa funzione restituisce un indice per l'azione migliore basato sull'osservazione
    return int(observation % nA)  # Ad esempio, scegli un'azione basata su un calcolo modulare
"""

def random_policy(nA):
    #na = numero azioni possibili
    def policy_fn(observation):
        # observation = stato attuale dell'ambiente
        # ma non viene utilizato
        #restituisce un array con una distribuzione di probabilita uniforme per scegliere l'azione

        return np.ones(nA, dtype=float) / nA

    return policy_fn


def epsilon_greedy_policy(Q, epsilon=0.1):
    def policy_fn(state):
        nA = len(Q[str(state)])
        A = np.ones(nA, dtype=float) * epsilon / (nA - 1)
        best_action = np.argmax(Q[str(state)])
        A[best_action] = 1.0 - epsilon
        A /= A.sum()
        return A

    return policy_fn


def encoding_action(action):
    if action==0:
        return "UP"
    elif action==1:
        return "RIGHT"
    elif action==2:
        return "LEFT"
    else:
        return "DOWN"


class MC_offpolicy:
    def __init__(self, env, num_episodes, discount_factor=0.99, epsilon=0.05):
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.C = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.behavior_policy = random_policy(self.env.action_space.n)
        self.target_policy = epsilon_greedy_policy(self.Q, epsilon=self.epsilon)

        # Clamping parameters
        self.max_q_value = 1e10
        self.min_q_value = -1e10
        self.epsilon_clamp = 1e-10

    def update(self, episode):
        #Si aggiorna Q sulla base dell'episodio preso in ingresso
        # G = totale delle ricompense dall'inizio alla fine dell'episodio
        G = 0.0
        #Peso per considerare il fatto che l'episodio
        # è stato generato da una policy differente
        W = 1.0

        for t in range(len(episode))[::-1]:
            #Ciclo dalla fine all'inizio stile MC
            state, action, reward = episode[t]
            G = self.discount_factor * G + reward
            self.C[str(state)][action] += 1

            old_value = self.Q[str(state)][action]

            weight = min(max(W, self.epsilon_clamp), 1e10)
            update_value = (weight / (self.C[str(state)][action] + self.epsilon_clamp)) * (G - old_value)

            new_value = old_value + update_value
            new_value = np.clip(new_value, self.min_q_value, self.max_q_value)
            self.Q[str(state)][action] = new_value

            if action == np.argmax(self.target_policy(state)):
                W *= (1.0 - self.epsilon) / max(self.epsilon, self.epsilon_clamp)
            else:
                W *= self.epsilon / max(self.epsilon, self.epsilon_clamp)

    def train(self, verbose=True):
        b_rewards = []
        t_rewards = []
        episode_length = [] #lunghezza episodio
        mean_rewards = []
        max_totalReward = -float('inf')

        for trial in range(1, self.num_episodes + 1):
            #inizio loop, ogni iterazione è un episodio
            start_time_trial = time.time()
            print(f"Episodio numero: {trial}...")
            if trial % 100 == 0:
                print(f"\rEpisode {trial}/{self.num_episodes}.", end="")
                sys.stdout.flush()

            episode = []
            #l'ambiente viene resettato, si riparte dalla posizione iniziale
            state = self.env.reset()
            done = False
            # done serve per determinare se un episodio è terminato
            step = 0
            numero_azione = 0
            while not done:
                start_time_action = time.time()
                #print(f"Numero azione: {numero_azione}")
                self.env.render()
                probs = self.behavior_policy(state)
                # print(f"PROBS: {probs}")
                if not np.isclose(np.sum(probs), 1.0):
                    raise ValueError(f"Behavior policy probabilities do not sum to 1: {probs}")
                #azione decisa random
                action = np.random.choice(np.arange(len(probs)), p=probs)
                azione = encoding_action(action)
                # print(f"Azione: {azione}")
                #aggiornamento stato e reward + check se l'episodio è finito
                #print(self.env.step(action))
                # next_state, reward, done, _, _ = self.env.step(action)
                next_state, reward, done, info, sol = self.env.step(action)
                episode.append((state, action, reward))
                # print(f"\nEPISODE: {episode}")
                state = next_state
                step += 1
                # numero_azione += 1
                end_time_action = time.time()  # Fine timer per l'azione
                #print(f"Tempo impiegato per l'azione: {end_time_action - start_time_action} secondi")

            # print(f"numero azioni:{numero_azione}")
            #Aggiornamento della funzione Q sulla base dell'episodio corrente
            end_time_trial = time.time()  # Fine timer per il ciclo trial
            print(f"Tempo impiegato per l'episodio: {end_time_trial - start_time_trial} secondi")
            self.update(episode)
            total_reward = sum([reward for _, _, reward in episode])
            b_rewards.append(total_reward)
            episode_length.append(len(episode))
            t_rewards.append(total_reward)
            mean_rewards.append(np.mean(t_rewards[-100:]))

            if verbose and trial % 20 == 0:
                print(f'\n---- Trial {trial} ----')
                print(f'Mean(last 10 total rewards): {np.mean(t_rewards[-10:])}')

            mean_totalReward = np.mean(t_rewards[-10:])
            if mean_totalReward > max_totalReward:
                max_totalReward = mean_totalReward
                print(f'The weights are saved with total rewards: {mean_totalReward}')

        print(f'Trial {trial} Total Reward: {t_rewards[-1]}')
        self.env.close()

        # Plot performance analysis
        plt.figure(figsize=(10, 5))
        plt.plot(t_rewards, label='Total Reward', color='blue')
        plt.plot(mean_rewards, label='Mean Reward (last 100 episodes)', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Total Reward and Mean Reward (last 100 episodes)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot policy grid
        self.plot_policy_grid()

        # Return the final policy as well
        return self.Q

    def plot_policy_grid(self):
        # Define the grid size for CliffWalking-v0 (4 rows x 12 columns)
        grid_height = 4
        grid_width = 12

        # Create a grid for the policy
        policy_grid = np.zeros((grid_height, grid_width), dtype=int)

        # Populate the policy grid with the action that has the highest Q-value
        for state in range(grid_height * grid_width):
            action = np.argmax(self.Q[str(state)])
            policy_grid[state // grid_width, state % grid_width] = action

        # Create a plot for the policy
        plt.figure(figsize=(10, 5))
        plt.imshow(policy_grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Action')
        plt.title('Learned Policy Grid')
        plt.xlabel('Column')
        plt.ylabel('Row')

        # Map actions to colors
        action_labels = ['Left', 'Down', 'Right', 'Up']
        plt.xticks(ticks=np.arange(grid_width), labels=np.arange(grid_width))
        plt.yticks(ticks=np.arange(grid_height), labels=np.arange(grid_height))

        # Overlay action labels
        for i in range(grid_height):
            for j in range(grid_width):
                plt.text(j, i, action_labels[policy_grid[i, j]], ha='center', va='center', color='white')

        plt.tight_layout()
        plt.show()

    def render_policy(self, Q, episodes=10):
        print(f'Rendering {episodes} episodes using the learned policy')

        # Create a copy of the environment for rendering
        env_copy = gym.make("CliffWalking-v0", render_mode="human")
        state = env_copy.reset()
        total_rewards = []
        for _ in range(episodes):
            done = False
            episode_reward = 0
            while not done:
                # Choose action based on Q-values
                action = np.argmax(Q[str(state)])
                state, reward, done, _, _ = env_copy.step(action)
                episode_reward += reward
                env_copy.render()  # Render the environment
                time.sleep(0.1)  # Delay to make rendering visible
            total_rewards.append(episode_reward)
            state = env_copy.reset()
        print(f'Average reward over {episodes} episodes: {np.mean(total_rewards)}')
        env_copy.close()


# Example usage
env = gym.make("CliffWalking-v0", render_mode="human")
rl = MC_offpolicy(env, num_episodes=500, epsilon=0.05)
Q = rl.train(verbose=False)

# Render the final learned policy using Q-values
rl.render_policy(Q, episodes=100)

