import gym
import numpy as np
import matplotlib.pyplot as plt
from spinup import vpg_pytorch, trpo, ppo

def plot_results(vpg_results, trpo_results, npg_results):
    plt.plot(vpg_results, label="VPG")
    plt.plot(trpo_results, label="TRPO")
    plt.plot(npg_results, label="NPG")
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.title('VPG vs TRPO vs NPG on CartPole-v1')
    plt.legend()
    plt.show()

def run_experiment():
    # Environment
    env_name = 'CartPole-v1'

    # VPG Experiment
    print("Training VPG...")
    vpg_results = vpg_pytorch(env_fn=lambda: gym.make(env_name), 
                      epochs=50, 
                      steps_per_epoch=4000)

    # TRPO Experiment
    print("Training TRPO...")
    trpo_results = trpo(env_fn=lambda: gym.make(env_name), 
                        epochs=50, 
                        steps_per_epoch=4000)

    # NPG Experiment (using PPO with very small clipping to simulate NPG)
    print("Training NPG...")
    npg_results = ppo(env_fn=lambda: gym.make(env_name), 
                      epochs=50, 
                      steps_per_epoch=4000,
                      pi_lr=1e-2,  # Low learning rate for stability
                      clip_ratio=0.01,  # Very small clipping for NPG-like behavior
                      target_kl=0.01)  # Tight constraint on KL divergence
    
    # Extracting the average returns
    vpg_avg_return = [ep['AverageReturn'] for ep in vpg_results['logger'].epoch_dicts]
    trpo_avg_return = [ep['AverageReturn'] for ep in trpo_results['logger'].epoch_dicts]
    npg_avg_return = [ep['AverageReturn'] for ep in npg_results['logger'].epoch_dicts]

    # Plotting the results
    plot_results(vpg_avg_return, trpo_avg_return, npg_avg_return)

if __name__ == "__main__":
    run_experiment()
