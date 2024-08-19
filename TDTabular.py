import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

class SARSA_agent(object):
    def solve(self, env, total_episodes=1000, alpha=0.9, epsilon=0.1, render_interval=100):
        """
        Solve a given Mountain Car environment using Temporal Difference learning (SARSA).
        
        input: 
          - env {gym.Env} -- Mountain Car environment to solve
          - total_episodes {int} -- Number of episodes to run the algorithm
          - alpha {float or str} -- Learning rate or 'Episode Inverse Decay'
          - epsilon {float} -- Exploration rate for epsilon-greedy policy
          - render_interval {int} -- Interval at which to render the environment

        output:
          - policy {np.array} -- Optimal policy found to solve the given Mountain Car environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of rewards for each episode 
        """
        # Define the number of bins for each state dimension
        num_bins = [10, 10]  # Number of bins for each dimension of the state space

        # Initialize Q-table
        state_size = tuple(num_bins)  # Discrete state space size
        action_size = env.action_space.n
        Q = np.zeros(state_size + (action_size,))

        # Initialize policy
        policy = np.ones(state_size + (action_size,)) * (1.0 / action_size)

        values = []
        total_rewards = []

        # On-policy TD control: SARSA
        for n_episodes in tqdm(range(total_episodes), unit="episode", leave=False):
            total_reward = 0.0
            state = tuple(env.reset())  # Directly use the state as a tuple
            action = np.random.choice(range(action_size), p=policy[state])

            # Set alpha_t if using 'Episode Inverse Decay'
            alpha_t = 1 / (n_episodes + 1) if isinstance(alpha, str) and alpha == 'Episode Inverse Decay' else alpha

            done = False
            while not done:
                if n_episodes % render_interval == 0:
                    env.render()

                next_state, reward, done, _ = env.step(action)
                next_state = tuple(next_state)  # Convert the state to a tuple
                total_reward += reward

                next_action = np.random.choice(range(action_size), p=policy[next_state])

                # Update Q-table using SARSA update rule
                Q[state + (action,)] += alpha_t * (reward + env.gamma * Q[next_state + (next_action,)] - Q[state + (action,)])

                # Update policy using epsilon-greedy approach
                a_opt = np.argmax(Q[state, :])
                policy[state] = np.ones(action_size) * epsilon / action_size
                policy[state][a_opt] = 1 - epsilon + epsilon / action_size

                state, action = next_state, next_action

            values.append(np.sum(policy * Q, axis=1))
            total_rewards.append(total_reward)

        env.close()

        return policy, values, total_rewards

def main():
    env = gym.make('MountainCar-v0')

    agent = SARSA_agent()
    policy, values, total_rewards = agent.solve(env, total_episodes=1000, alpha=0.1, epsilon=0.1, render_interval=100)

    # Plot total rewards
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    # Print some results
    print(f"Average total reward: {np.mean(total_rewards)}")
    print(f"Best policy value: {np.max(np.mean(values, axis=0))}")

if __name__ == "__main__":
    main()
