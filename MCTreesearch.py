import numpy as np
import gymnasium as gym

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, action_size):
        return len(self.children) == action_size

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action, next_state):
        child_node = MCTSNode(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def fully_expanded_and_best_child(self, action_size, c_param=1.4):
        if self.is_fully_expanded(action_size):
            return self.best_child(c_param)
        return None

class MCTS:
    def __init__(self, env, num_simulations=1000):
        self.env = env
        self.num_simulations = num_simulations
        self.root = None

    def search(self, initial_state):
        self.root = MCTSNode(state=initial_state)

        for _ in range(self.num_simulations):
            node = self.root
            env_copy = gym.make(self.env.spec.id, render_mode="human")  # Create a fresh environment copy
            state = initial_state

            # Selection
            while node.is_fully_expanded(self.env.action_space.n) and len(node.children) > 0:
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded(self.env.action_space.n):
                action = self._select_untried_action(node)
                next_state, reward, done, _, _ = env_copy.step(action)
                if done:
                    node.update(reward)
                    continue
                node = node.expand(action, next_state)

            # Simulation
            reward = self._simulate(env_copy, node.state)

            # Backpropagation
            self._backpropagate(node, reward)

        # Return the best action from the root node
        return self.root.best_child(c_param=0.0).action

    def _select_untried_action(self, node):
        tried_actions = [child.action for child in node.children]
        possible_actions = set(range(self.env.action_space.n)) - set(tried_actions)
        return np.random.choice(list(possible_actions))

    def _simulate(self, env_copy, state):
        done = False
        total_reward = 0
        while not done:
            action = np.random.choice(self.env.action_space.n)
            state, reward, done, _, _ = env_copy.step(action)
            total_reward += reward
        return total_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

# Example usage with FrozenLake-v1 environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
mcts = MCTS(env, num_simulations=1000)
state, _ = env.reset()
done = False

while not done:
    action = mcts.search(state)
    state, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        print(f"Finished with reward: {reward}")

env.close()
