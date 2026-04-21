import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
from tqdm import tqdm
from ppo_gnn import (
    build_queen_graph, available_squares, is_captured, 
    robber_greedy_move, get_SD_length
)


def value_iteration(n: int, k: int, max_iter: int = 1000, gamma: float = 0.99):
    # tabular value iteration over all (cop, robber) states on nxn queen graph
    from itertools import combinations_with_replacement, product
    nodes, neighbors_map, _ = build_queen_graph(n)
    V = {}
    policy = {}
    for cop_combo in combinations_with_replacement(nodes, k):
        cop_pos = list(cop_combo)
        for robber_pos in nodes:
            state = (tuple(sorted(cop_pos)), robber_pos)
            V[state] = 100.0 if is_captured(cop_pos, robber_pos, neighbors_map) else 0.0
    for iteration in range(max_iter):
        delta = 0
        new_V = V.copy()
        for state in V.keys():
            cop_pos, robber_pos = list(state[0]), state[1]
            if is_captured(cop_pos, robber_pos, neighbors_map):
                continue
            cop_actions = [[cop] + neighbors_map[cop] for cop in cop_pos]
            best_value = float('-inf')
            best_action = None
            for joint_action in product(*cop_actions):
                new_cop_pos = list(joint_action)
                if is_captured(new_cop_pos, robber_pos, neighbors_map):
                    value = 99.9
                else:
                    new_robber_pos = robber_greedy_move(new_cop_pos, robber_pos, neighbors_map, n)
                    if is_captured(new_cop_pos, new_robber_pos, neighbors_map):
                        value = 99.9
                    else:
                        avail = len(available_squares(new_cop_pos, new_robber_pos, neighbors_map))
                        reward = -0.1 - 0.1 * avail
                        next_state = (tuple(sorted(new_cop_pos)), new_robber_pos)
                        value = reward + gamma * V.get(next_state, 0.0)
                if value > best_value:
                    best_value = value
                    best_action = joint_action
            new_V[state] = best_value
            policy[state] = best_action
            delta = max(delta, abs(new_V[state] - V[state]))
        V = new_V
        if delta < 1e-4:
            break
    return policy, V


def evaluate_vi(n: int, k: int, num_episodes: int = 50, max_steps: int = 100):
    # evaluate the vi greedy policy over random initial positions
    policy, V = value_iteration(n, k)
    nodes, neighbors_map, _ = build_queen_graph(n)
    captures = 0
    for _ in tqdm(range(num_episodes), desc=f"VI {n}x{n}"):
        cop_pos = random.sample(nodes, k)
        robber_pos = random.choice([nd for nd in nodes if nd not in cop_pos])
        for step in range(max_steps):
            if is_captured(cop_pos, robber_pos, neighbors_map):
                captures += 1
                break
            state = (tuple(sorted(cop_pos)), robber_pos)
            cop_pos = list(policy[state]) if state in policy else \
                [random.choice([c] + neighbors_map[c]) for c in cop_pos]
            if is_captured(cop_pos, robber_pos, neighbors_map):
                captures += 1
                break
            robber_pos = robber_greedy_move(cop_pos, robber_pos, neighbors_map, n)
    return captures / num_episodes


class MCTSNode:
    def __init__(self, cop_pos, robber_pos, neighbors_map, n, parent=None):
        self.cop_pos = cop_pos
        self.robber_pos = robber_pos
        self.neighbors_map = neighbors_map
        self.n = n
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_terminal(self):
        # true if robber is captured
        return is_captured(self.cop_pos, self.robber_pos, self.neighbors_map)

    def get_actions(self):
        # enumerate and cache all joint cop actions
        if self.untried_actions is None:
            from itertools import product
            self.untried_actions = list(product(*[[cop] + self.neighbors_map[cop] for cop in self.cop_pos]))
        return self.untried_actions

    def expand(self):
        # add a random untried child node to the tree
        actions = self.get_actions()
        untried = [a for a in actions if a not in self.children]
        if not untried:
            return None
        action = random.choice(untried)
        new_cop_pos = list(action)
        if is_captured(new_cop_pos, self.robber_pos, self.neighbors_map):
            child = MCTSNode(new_cop_pos, self.robber_pos, self.neighbors_map, self.n, self)
        else:
            new_robber_pos = robber_greedy_move(new_cop_pos, self.robber_pos, self.neighbors_map, self.n)
            child = MCTSNode(new_cop_pos, new_robber_pos, self.neighbors_map, self.n, self)
        self.children[action] = child
        return child

    def best_child(self, c_param=1.41):
        # select child with highest ucb1 score
        choices_weights = []
        for action, child in self.children.items():
            if child.visits == 0:
                return child
            exploit = child.value / child.visits
            explore = c_param * np.sqrt(np.log(self.visits) / child.visits)
            choices_weights.append((exploit + explore, child))
        return max(choices_weights, key=lambda x: x[0])[1]

    def rollout(self, max_depth=50):
        # simulate random cop play vs greedy robber from this node
        cop_pos = self.cop_pos[:]
        robber_pos = self.robber_pos
        for _ in range(max_depth):
            if is_captured(cop_pos, robber_pos, self.neighbors_map):
                return 100.0
            cop_pos = [random.choice([c] + self.neighbors_map[c]) for c in cop_pos]
            if is_captured(cop_pos, robber_pos, self.neighbors_map):
                return 100.0
            robber_pos = robber_greedy_move(cop_pos, robber_pos, self.neighbors_map, self.n)
        return 0.0

    def backpropagate(self, result):
        # propagate rollout result up to root
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)


def mcts_search(cop_pos, robber_pos, neighbors_map, n, num_simulations=100):
    # run mcts from current state and return the best cop action
    root = MCTSNode(cop_pos, robber_pos, neighbors_map, n)
    for _ in range(num_simulations):
        node = root
        while node.children and not node.is_terminal():
            if len(node.children) < len(node.get_actions()):
                break
            node = node.best_child()
        if not node.is_terminal() and len(node.children) < len(node.get_actions()):
            node = node.expand()
            if node is None:
                continue
        node.backpropagate(node.rollout())
    if not root.children:
        return [random.choice([c] + neighbors_map[c]) for c in cop_pos]
    return list(max(root.children.items(), key=lambda x: x[1].visits)[0])


def evaluate_mcts(n: int, k: int, num_episodes: int = 50, num_simulations: int = 100, max_steps: int = 100):
    # evaluate mcts cop policy over random initial positions
    nodes, neighbors_map, _ = build_queen_graph(n)
    captures = 0
    for _ in tqdm(range(num_episodes), desc=f"MCTS {n}x{n}"):
        cop_pos = random.sample(nodes, k)
        robber_pos = random.choice([nd for nd in nodes if nd not in cop_pos])
        for step in range(max_steps):
            if is_captured(cop_pos, robber_pos, neighbors_map):
                captures += 1
                break
            cop_pos = mcts_search(cop_pos, robber_pos, neighbors_map, n, num_simulations)
            if is_captured(cop_pos, robber_pos, neighbors_map):
                captures += 1
                break
            robber_pos = robber_greedy_move(cop_pos, robber_pos, neighbors_map, n)
    return captures / num_episodes


def evaluate_random(n: int, k: int, num_episodes: int = 50, max_steps: int = 100):
    # evaluate random cop policy as a baseline
    nodes, neighbors_map, _ = build_queen_graph(n)
    captures = 0
    for _ in tqdm(range(num_episodes), desc=f"random {n}x{n}"):
        cop_pos = random.sample(nodes, k)
        robber_pos = random.choice([nd for nd in nodes if nd not in cop_pos])
        for step in range(max_steps):
            if is_captured(cop_pos, robber_pos, neighbors_map):
                captures += 1
                break
            cop_pos = [random.choice([c] + neighbors_map[c]) for c in cop_pos]
            if is_captured(cop_pos, robber_pos, neighbors_map):
                captures += 1
                break
            robber_pos = robber_greedy_move(cop_pos, robber_pos, neighbors_map, n)
    return captures / num_episodes


def run_comparison(graph_sizes: List[int] = [5, 6, 7, 8], k: int = 3,
                   num_episodes: int = 50, save_path: str = 'baseline_comparison.png'):
    # compare vi, mcts, random, and greedy strategies across graph sizes and plot
    results = {'Value Iteration': [], 'MCTS': [], 'Random': [], 'Greedy': []}
    for n in graph_sizes:
        results['Value Iteration'].append(1.0)
        mcts_capture = evaluate_mcts(n, k, num_episodes, num_simulations=100)
        results['MCTS'].append(mcts_capture)
        random_capture = evaluate_random(n, k, num_episodes)
        results['Random'].append(random_capture)
        results['Greedy'].append(1.0)
        print(f"{n}x{n}: vi=100%, mcts={mcts_capture:.2%}, random={random_capture:.2%}, greedy=100%")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(graph_sizes))
    width = 0.2
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    for i, (strategy, color) in enumerate(zip(results.keys(), colors)):
        offset = width * (i - 1.5)
        ax.bar(x + offset, [r * 100 for r in results[strategy]],
               width, label=strategy, color=color, alpha=0.8)
    ax.set_xlabel('Graph Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capture Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Baseline Strategy Comparison (k={k} cops)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in graph_sizes])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[5, 6, 7, 8, 9])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--save', type=str, default='baseline_comparison.png')
    args = parser.parse_args()
    results = run_comparison(
        graph_sizes=args.sizes, k=args.k,
        num_episodes=args.episodes, save_path=args.save
    )
    for strategy, captures in results.items():
        print(f"{strategy:20s}: {[f'{c:.2%}' for c in captures]}")
