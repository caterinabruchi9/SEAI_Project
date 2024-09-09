# Monte Carlo Methods for Policy Optimization

## Overview

This paper provides a comprehensive overview of Monte Carlo methods for Policy Optimization tested on the Gymnasium Frozen Lake environment, offering an alternative to Temporal Difference (TD) methods. Monte Carlo methods are advantageous in scenarios where complete knowledge of the environment is not required. Instead of needing a full probability distribution of all possible transitions, Monte Carlo methods leverage experience—sample sequences of states, actions, and rewards—from interactions with an environment. This approach avoids complexities associated with dynamic programming in complex environments.

## Key Features of Monte Carlo Methods

- **Model-Free**: Eliminates the need for a well-known mathematical model of the environment.
- **Sample-Based**: Operates without considering all states and actions simultaneously.
- **Off-Line Learning**: Learns from experiences gathered at the end of episodes rather than during agent behavior.
- **Unbiased Learning**: Computes expected returns using the average of returns.
- **High Variance**: Returns can vary significantly until the end of the episode, unlike TD methods which reduce long-term variability through one-step learning.


## Tabular Methods

The overview begins with the simplest Monte Carlo method: **tabular** methods. These methods are effective for small learning spaces, as they store and update values for state-action pairs in a table, providing accurate estimates of expected returns. 

In a typical Monte Carlo implementation, a **Q-Table** is realized as a \(16 x 4\) matrix, where:
- **Columns** represent possible actions.
- **Rows** represent possible states.
- Each entry in the matrix represents the reward achieved from each state-action pair.

The Q-Table is updated only after an episode terminates. By the end of the training phase, higher matrix values indicate the optimal path from the starting point to the goal.

#### Policy Optimization
Policy optimization algorithms can be categorized as **ON-policy** or **OFF-policy**:

- **ON-policy**: In this approach, episodes are collected using the same policy that is being optimized. The learning process directly improves the policy based on the experiences gathered under that policy.
- **OFF-policy**: Here, experience is collected using an exploratory policy \(\mu\), different from the policy being optimized. This allows learning from a wider range of experiences, not limited to the actions taken by the current policy.

Both ON-policy and OFF-policy methods have been implemented for the analysis of this tabular method.

### Update Strategies

Monte Carlo approaches update the Q-Table in one of two ways:

- **First Visit**: The entry in the table is updated only the first time the state-action pair is visited in an episode.
- **Every Visit**: The entry in the table is updated every time the state-action pair is visited during an episode, computing the average reward.

## Monte Carlo Tree Search

A more advanced application of Monte Carlo methods is the **Monte Carlo Tree Search (MCTS)**. MCTS is a successful decision-time planning technique that builds a search tree using simulated samples in the search space. The core idea involves updating the set of visited states through four steps:

1. **Selection**: Start from the root node and select not-fully expanded children nodes in subsequent iterations.
2. **Expansion**: Add new child nodes by selecting actions that have not been tested yet, unless a termination condition is met.
3. **Playout (or Rollout)**: Simulate from the selected node until the termination state is reached.
4. **Backpropagation**: Update the number of visits (`N(s,a)`) and rewards (`Q(s,a)`) for state-action pairs based on the received reward (`r`).

Rollout algorithms are used to estimate action values by averaging the returns of many simulated trajectories starting from the current state and following a given policy. Initially, MCTS aimed to implement a full tree with simulated paths to terminal states. However, for complex games, this approach proved unsuitable, leading to more effective solutions using value functions and policy approximations.


### Monte Carlo Tree Search (MCTS) Variants

Starting from the Naive version, many adaptations of the MCTS algorithm were developed to address different challenges and improve performance. The UCB variant was previously compared to the greedy approach, but these are not the only methods to address the issue of tree size, which can make computations heavier.

Overall, we have analyzed several versions of MCTS, including:

### Simpler and Full Tree Versions
- **Naive MCTS \(\epsilon\)-greedy**: This version uses a simple \(\epsilon\)-greedy strategy to balance exploration and exploitation.
- **Naive MCTS UCT**: This version incorporates Upper Confidence Bounds for Trees (UCT) to guide exploration and improve performance.

### More Efficient Versions
- **Bootstrapped MCTS**: This variant utilizes model knowledge, specifically leveraging the value function \(V\), to enhance efficiency.
- **Bootstrapped MCTS with UCT**: This version combines Bootstrapped MCTS with UCT to further refine exploration strategies and improve decision-making.

In addition to these versions, we will introduce variability in the environment's behavior by implementing a custom slippery function. This function will add more robustness to our results by simulating a less deterministic environment.

For didactic purposes, the algorithms have been tested in a simple environment, which is also suitable for much simpler algorithms like the tabular methods described earlier.

