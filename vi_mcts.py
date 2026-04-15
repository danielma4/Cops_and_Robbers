"""
Value Iteration & Monte Carlo Tree Search for Cops and Robbers on Queen Graphs.

Classical planning algorithms applied to the cops-and-robbers game.
"""

import numpy as np
import random
import math
import math as _math
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from sage.all import graphs
from tqdm import tqdm
from itertools import combinations


# =============================================================================
# Shared Game Helpers
# =============================================================================

def build_graph_data(n):
    """Build queen graph and neighbor lookup for an nxn board."""
    G = graphs.QueenGraph([n, n])
    nodes = [(int(a), int(b)) for (a, b) in G.vertices()]
    neighbors_map = {}
    for v in nodes:
        neighbors_map[v] = [(int(u[0]), int(u[1])) for u in G.neighbors(v)]
    return G, nodes, neighbors_map


def available_squares(cop_pos, robber_pos, neighbors_map):
    """
    Return the set of squares available to the robber.
    Cops block their position AND all adjacent squares (domination).
    Robber can move to a neighbor or stay put, minus blocked squares.
    """
    blocked = set(cop_pos)
    for cop in cop_pos:
        blocked.update(neighbors_map[cop])
    if robber_pos == -1:
        return [v for v in neighbors_map if v not in blocked]
    moves = set(neighbors_map[robber_pos]) | {robber_pos}
    return [m for m in moves if m not in blocked]


def is_captured(cop_pos, robber_pos, neighbors_map):
    return len(available_squares(cop_pos, robber_pos, neighbors_map)) == 0


def get_cop_actions(cops, neighbors_map):
    """
    Generate all joint cop actions: each cop independently stays or moves
    to any neighbor, producing all combinations.
    """
    options = []
    for c in cops:
        options.append([c] + neighbors_map[c])
    joint = set()
    for combo in product(*options):
        joint.add(tuple(sorted(combo)))
    return list(joint)


def get_SD_length(robber_pos, n):
    """Length of the shortest diagonal through robber_pos on an nxn board."""
    r, c = robber_pos
    pos_diag = sum(1 for rr in range(n) for cc in range(n) if rr - cc == r - c)
    neg_diag = sum(1 for rr in range(n) for cc in range(n) if rr + cc == r + c)
    return min(pos_diag, neg_diag)


def robber_greedy_move(cop_pos, robber_pos, neighbors_map, n):
    """
    Robber greedy strategy: pick the available move that maximizes
    SD length, breaking ties by number of available squares.
    """
    avail = available_squares(cop_pos, robber_pos, neighbors_map)
    if not avail:
        return robber_pos
    return max(avail, key=lambda m: (
        get_SD_length(m, n),
        len(available_squares(cop_pos, m, neighbors_map))
    ))


# =============================================================================
# Value Iteration
# =============================================================================

def value_iteration_cops_robbers(n, k=3, gamma=0.99, theta=1e-6, max_iters=500):
    """
    Tabular value iteration on an nxn queen graph with k cops vs 1 robber.

    Game rules (matching strat_v5):
      - Cops dominate: block their square AND all neighbors
      - All cops move simultaneously (any subset may move)
      - Robber follows greedy strategy: maximize SD length
      - Game ends when robber has 0 available squares
    """

    G, nodes, neighbors_map = build_graph_data(n)

    def step_reward(cop_pos, robber_pos, captured):
        if captured:
            return 100.0
        sd = get_SD_length(robber_pos, n)
        return -0.1 - sd

    # Enumerate or sample states
    num_total = len(list(combinations(nodes, k))) * len(nodes)
    if num_total <= 100_000:
        state_list = [
            (cops_t, rob)
            for cops_t in combinations(nodes, k)
            for rob in nodes
        ]
    else:
        num_sampled = min(50000, num_total)
        state_set = set()
        while len(state_set) < num_sampled:
            cops = tuple(sorted(random.sample(nodes, k)))
            rob = random.choice(nodes)
            state_set.add((cops, rob))
        state_list = list(state_set)

    print(f"  {len(state_list)} states to sweep over")

    V = defaultdict(float)
    td_errors_per_iter = []

    for iteration in range(max_iters):
        delta = 0.0
        td_errors = []
        for (cops, rob) in tqdm(state_list,
                                desc=f"  VI iter {iteration+1}/{max_iters}",
                                leave=False):
            if is_captured(cops, rob, neighbors_map):
                V[(cops, rob)] = 100.0
                continue
            old_v = V[(cops, rob)]

            # Bellman backup: max over cop joint actions
            # Robber uses greedy strategy (deterministic), not minimax
            best_val = -1e9
            for new_cops_t in get_cop_actions(cops, neighbors_map):
                if is_captured(new_cops_t, rob, neighbors_map):
                    val = step_reward(new_cops_t, rob, True) + gamma * 100.0
                else:
                    rob_next = robber_greedy_move(new_cops_t, rob, neighbors_map, n)
                    val = step_reward(new_cops_t, rob, False) + gamma * V[(new_cops_t, rob_next)]
                best_val = max(best_val, val)

            V[(cops, rob)] = best_val
            td = abs(best_val - old_v)
            td_errors.append(td)
            delta = max(delta, td)

        mean_td = np.mean(td_errors)
        td_errors_per_iter.append(mean_td)
        print(f"  Iter {iteration+1}: delta={delta:.6f}, mean_td={mean_td:.6f}")
        if delta < theta:
            print(f"  VI converged at iteration {iteration + 1}")
            break

    # Evaluate the greedy policy with rollouts
    num_eval = 200
    max_ep_len = max(n * n, 100)
    captures = 0
    episode_lengths = []
    avg_rewards = []
    value_estimates = []
    actual_returns = []

    for ep in tqdm(range(num_eval), desc="  VI eval"):
        cops = tuple(sorted(random.sample(nodes, k)))
        rob = robber_greedy_move(cops, -1, neighbors_map, n)  # robber picks best start
        ep_reward = 0.0
        v_est = V[(cops, rob)]

        for t in range(max_ep_len):
            if is_captured(cops, rob, neighbors_map):
                ep_reward += 100.0
                captures += 1
                episode_lengths.append(t)
                break

            # Greedy cop action from V
            best_val = -1e9
            best_cops = cops
            for new_cops_t in get_cop_actions(cops, neighbors_map):
                if is_captured(new_cops_t, rob, neighbors_map):
                    val = 1e9
                else:
                    rob_next = robber_greedy_move(new_cops_t, rob, neighbors_map, n)
                    val = V[(new_cops_t, rob_next)]
                if val > best_val:
                    best_val = val
                    best_cops = new_cops_t
            cops = best_cops

            r = step_reward(cops, rob, is_captured(cops, rob, neighbors_map))
            ep_reward += r

            if is_captured(cops, rob, neighbors_map):
                captures += 1
                episode_lengths.append(t + 1)
                break

            rob = robber_greedy_move(cops, rob, neighbors_map, n)
        else:
            episode_lengths.append(max_ep_len)

        avg_rewards.append(ep_reward)
        value_estimates.append(v_est)
        actual_returns.append(ep_reward)

    capture_rate = captures / num_eval

    return {
        'td_errors': td_errors_per_iter,
        'episode_lengths': episode_lengths,
        'capture_rate': capture_rate,
        'avg_rewards': avg_rewards,
        'value_estimates': value_estimates,
        'actual_returns': actual_returns,
        'n': n,
        'k': k,
        'V': V,
        'neighbors_map': neighbors_map,
        'nodes': nodes,
    }


def run_vi_sweep(sizes, k=3, **kwargs):
    """Run value iteration across multiple queen graph sizes and return all metrics."""
    all_metrics = []
    for n in sizes:
        print(f"Running VI on {n}x{n} queen graph (k={k})...")
        m = value_iteration_cops_robbers(n, k=k, **kwargs)
        all_metrics.append(m)
        print(f"  Capture rate: {m['capture_rate']:.2%}, "
              f"Avg episode length: {np.mean(m['episode_lengths']):.1f}")
    return all_metrics


# =============================================================================
# Monte Carlo Tree Search
# =============================================================================

class MCTSNode:
    """A node in the MCTS game tree (cop turn only -- robber is deterministic)."""
    __slots__ = ['cops', 'rob', 'parent', 'children',
                 'visits', 'value', 'untried_actions', '_neighbors_map', '_n']

    def __init__(self, cops, rob, parent, neighbors_map, n):
        self.cops = cops
        self.rob = rob
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self._neighbors_map = neighbors_map
        self._n = n

    def is_terminal(self):
        return is_captured(self.cops, self.rob, self._neighbors_map)

    def get_actions(self):
        """Sample joint cop actions (full product is too large, so sample)."""
        if self.untried_actions is not None:
            return self.untried_actions
        # For each cop, sample a subset of moves to keep branching manageable
        options = []
        for c in self.cops:
            cands = [c] + self._neighbors_map[c]
            options.append(random.sample(cands, min(len(cands), 5)))
        actions = list(set(tuple(sorted(combo)) for combo in product(*options)))
        random.shuffle(actions)
        self.untried_actions = actions[:50]  # cap branching factor
        return self.untried_actions

    def best_child(self, c=1.41):
        """UCB1 selection (cops maximizing)."""
        best_score = -1e9
        best = None
        for action, child in self.children.items():
            if child.visits == 0:
                return child
            exploit = child.value / child.visits
            explore = c * _math.sqrt(_math.log(self.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best


def mcts_cops_robbers(n, k=3, num_simulations=300, num_eval_episodes=200,
                      max_ep_len=None, rollout_depth=50):
    """
    Run MCTS-based cop policy on an nxn queen graph.

    Game rules (matching strat_v5):
      - Cops dominate their neighborhoods
      - All cops move simultaneously
      - Robber follows greedy SD-maximizing strategy
      - Game ends when robber has 0 available squares
    """
    G, nodes, neighbors_map = build_graph_data(n)

    if max_ep_len is None:
        max_ep_len = max(n * n, 100)

    def random_rollout(cops, rob, depth):
        """Random cop rollout vs greedy robber."""
        total_r = 0.0
        for _ in range(depth):
            if is_captured(cops, rob, neighbors_map):
                total_r += 100.0
                break
            # Random joint cop move
            new_cops = list(cops)
            for ci in range(k):
                cands = [cops[ci]] + neighbors_map[cops[ci]]
                new_cops[ci] = random.choice(cands)
            cops = tuple(sorted(new_cops))

            if is_captured(cops, rob, neighbors_map):
                total_r += 100.0
                break

            # Greedy robber response
            rob = robber_greedy_move(cops, rob, neighbors_map, n)
            total_r -= 0.1
        return total_r

    def mcts_action(cops, rob):
        """Run MCTS and return best cop configuration."""
        root = MCTSNode(cops, rob, None, neighbors_map, n)

        for _ in range(num_simulations):
            node = root

            # Selection
            while node.untried_actions is not None and len(node.untried_actions) == 0 and node.children:
                node = node.best_child()

            # Expansion
            actions = node.get_actions()
            untried = [a for a in actions if a not in node.children]
            if untried and not node.is_terminal():
                action = random.choice(untried)
                # After cops move, robber responds deterministically
                if is_captured(action, node.rob, neighbors_map):
                    rob_next = node.rob
                else:
                    rob_next = robber_greedy_move(action, node.rob, neighbors_map, n)
                child = MCTSNode(action, rob_next, node, neighbors_map, n)
                node.children[action] = child
                node.untried_actions = [a for a in actions if a not in node.children]
                node = child

            # Simulation (rollout)
            reward = random_rollout(node.cops, node.rob, rollout_depth)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        if not root.children:
            return cops
        best_action = max(root.children, key=lambda a: root.children[a].visits)
        return best_action

    # Evaluation rollouts
    captures = 0
    episode_lengths = []
    avg_rewards = []

    from tqdm import tqdm as _tqdm
    for ep in _tqdm(range(num_eval_episodes), desc=f"  MCTS eval {n}x{n}"):
        cops = tuple(sorted(random.sample(nodes, k)))
        rob = robber_greedy_move(cops, -1, neighbors_map, n)  # robber picks best start
        ep_reward = 0.0

        for t in range(max_ep_len):
            if is_captured(cops, rob, neighbors_map):
                ep_reward += 100.0
                captures += 1
                episode_lengths.append(t)
                break

            cops = mcts_action(cops, rob)

            if is_captured(cops, rob, neighbors_map):
                ep_reward += 100.0
                captures += 1
                episode_lengths.append(t + 1)
                break

            rob = robber_greedy_move(cops, rob, neighbors_map, n)
            ep_reward -= 0.1
        else:
            episode_lengths.append(max_ep_len)

        avg_rewards.append(ep_reward)

    capture_rate = captures / num_eval_episodes
    return {
        'episode_lengths': episode_lengths,
        'capture_rate': capture_rate,
        'avg_rewards': avg_rewards,
        'n': n,
        'k': k,
        'mcts_action': mcts_action,
        'neighbors_map': neighbors_map,
        'nodes': nodes,
    }


def run_mcts_sweep(sizes, k=3, num_simulations=200, num_eval_episodes=100, **kwargs):
    """Run MCTS across multiple queen graph sizes."""
    all_metrics = []
    for n in sizes:
        print(f"Running MCTS on {n}x{n} queen graph (k={k}, sims={num_simulations})...")
        m = mcts_cops_robbers(n, k=k, num_simulations=num_simulations,
                              num_eval_episodes=num_eval_episodes, **kwargs)
        all_metrics.append(m)
        print(f"  Capture rate: {m['capture_rate']:.2%}, "
              f"Avg episode length: {np.mean(m['episode_lengths']):.1f}")
    return all_metrics


# =============================================================================
# Metrics Tracking & Plotting
# =============================================================================

def plot_vi_metrics(all_metrics):
    """
    Plot metrics from value iteration runs across graph sizes.
    Includes: capture rate, episode length, avg reward, TD error, V(s) vs return.
    """
    sizes = [m['n'] for m in all_metrics]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Value Iteration -- Metrics across Queen Graph Sizes', fontsize=14)

    # 1) Capture rate vs graph size
    ax = axes[0, 0]
    cap_rates = [m['capture_rate'] for m in all_metrics]
    ax.bar(range(len(sizes)), cap_rates, tick_label=[f'{s}x{s}' for s in sizes], color='steelblue')
    ax.set_ylabel('Capture Rate')
    ax.set_xlabel('Graph Size')
    ax.set_title('Capture Rate vs Graph Size')
    ax.set_ylim(0, 1.05)
    for i, cr in enumerate(cap_rates):
        ax.text(i, cr + 0.02, f'{cr:.0%}', ha='center', fontsize=9)

    # 2) Average episode length vs graph size
    ax = axes[0, 1]
    avg_lens = [np.mean(m['episode_lengths']) for m in all_metrics]
    ax.bar(range(len(sizes)), avg_lens, tick_label=[f'{s}x{s}' for s in sizes], color='coral')
    ax.set_ylabel('Avg Episode Length')
    ax.set_xlabel('Graph Size')
    ax.set_title('Avg Episode Length vs Graph Size')

    # 3) Average reward vs graph size
    ax = axes[0, 2]
    avg_rews = [np.mean(m['avg_rewards']) for m in all_metrics]
    ax.bar(range(len(sizes)), avg_rews, tick_label=[f'{s}x{s}' for s in sizes], color='mediumseagreen')
    ax.set_ylabel('Avg Reward')
    ax.set_xlabel('Graph Size')
    ax.set_title('Avg Reward vs Graph Size')

    # 4) TD error convergence curves
    ax = axes[1, 0]
    for m in all_metrics:
        ax.plot(m['td_errors'], label=f"{m['n']}x{m['n']}")
    ax.set_ylabel('Mean Absolute TD Error')
    ax.set_xlabel('Iteration')
    ax.set_title('TD Error Convergence')
    ax.legend()
    ax.set_yscale('log')

    # 5) Value estimate vs actual return (scatter)
    ax = axes[1, 1]
    for m in all_metrics:
        ax.scatter(m['value_estimates'], m['actual_returns'],
                   alpha=0.3, s=10, label=f"{m['n']}x{m['n']}")
    lims = ax.get_xlim()
    ax.plot(lims, lims, 'k--', alpha=0.5, label='ideal')
    ax.set_xlabel('Value Estimate V(s0)')
    ax.set_ylabel('Actual Return')
    ax.set_title('Value Estimate vs Actual Return')
    ax.legend(fontsize=8)

    # 6) Episode length distribution (box plot)
    ax = axes[1, 2]
    ax.boxplot([m['episode_lengths'] for m in all_metrics],
               labels=[f'{s}x{s}' for s in sizes])
    ax.set_ylabel('Episode Length')
    ax.set_xlabel('Graph Size')
    ax.set_title('Episode Length Distribution')

    plt.tight_layout()
    plt.show()


def plot_mcts_metrics(all_metrics):
    """
    Plot metrics from MCTS runs across graph sizes.
    Note: MCTS has no entropy or TD error -- only capture rate, episode length, reward.
    """
    sizes = [m['n'] for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MCTS -- Metrics across Queen Graph Sizes', fontsize=14)

    # Capture rate
    ax = axes[0]
    cap_rates = [m['capture_rate'] for m in all_metrics]
    ax.bar(range(len(sizes)), cap_rates, tick_label=[f'{s}x{s}' for s in sizes], color='steelblue')
    ax.set_ylabel('Capture Rate')
    ax.set_xlabel('Graph Size')
    ax.set_title('Capture Rate vs Graph Size')
    ax.set_ylim(0, 1.05)
    for i, cr in enumerate(cap_rates):
        ax.text(i, cr + 0.02, f'{cr:.0%}', ha='center', fontsize=9)

    # Episode length
    ax = axes[1]
    avg_lens = [np.mean(m['episode_lengths']) for m in all_metrics]
    ax.bar(range(len(sizes)), avg_lens, tick_label=[f'{s}x{s}' for s in sizes], color='coral')
    ax.set_ylabel('Avg Episode Length')
    ax.set_xlabel('Graph Size')
    ax.set_title('Avg Episode Length vs Graph Size')

    # Avg reward
    ax = axes[2]
    avg_rews = [np.mean(m['avg_rewards']) for m in all_metrics]
    ax.bar(range(len(sizes)), avg_rews, tick_label=[f'{s}x{s}' for s in sizes], color='mediumseagreen')
    ax.set_ylabel('Avg Reward')
    ax.set_xlabel('Graph Size')
    ax.set_title('Avg Reward vs Graph Size')

    plt.tight_layout()
    plt.show()


def plot_comparison(vi_metrics, mcts_metrics):
    """Side-by-side comparison of VI vs MCTS on capture rate, episode length, avg reward."""
    vi_sizes = [m['n'] for m in vi_metrics]
    mcts_sizes = [m['n'] for m in mcts_metrics]
    common = sorted(set(vi_sizes) & set(mcts_sizes))

    if not common:
        print("No common graph sizes to compare.")
        return

    vi_map = {m['n']: m for m in vi_metrics}
    mcts_map = {m['n']: m for m in mcts_metrics}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Value Iteration vs MCTS -- Comparison', fontsize=14)
    x = np.arange(len(common))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, [vi_map[s]['capture_rate'] for s in common], w, label='VI', color='steelblue')
    ax.bar(x + w/2, [mcts_map[s]['capture_rate'] for s in common], w, label='MCTS', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in common])
    ax.set_ylabel('Capture Rate')
    ax.set_title('Capture Rate')
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.bar(x - w/2, [np.mean(vi_map[s]['episode_lengths']) for s in common], w, label='VI', color='steelblue')
    ax.bar(x + w/2, [np.mean(mcts_map[s]['episode_lengths']) for s in common], w, label='MCTS', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in common])
    ax.set_ylabel('Avg Episode Length')
    ax.set_title('Avg Episode Length')
    ax.legend()

    ax = axes[2]
    ax.bar(x - w/2, [np.mean(vi_map[s]['avg_rewards']) for s in common], w, label='VI', color='steelblue')
    ax.bar(x + w/2, [np.mean(mcts_map[s]['avg_rewards']) for s in common], w, label='MCTS', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in common])
    ax.set_ylabel('Avg Reward')
    ax.set_title('Avg Reward')
    ax.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# Game Simulation & Visualization
# =============================================================================

import networkx as nx

def simulate_vi_game(vi_result, cop_start=None, robber_start=None, max_steps=None):
    """
    Simulate a single game using the greedy VI policy.
    Returns (cop_moves, robber_moves) as lists of states per turn.
    """
    V = vi_result['V']
    nodes = vi_result['nodes']
    n = vi_result['n']
    k = vi_result['k']
    nmap = vi_result['neighbors_map']

    if max_steps is None:
        max_steps = max(n * n, 100)

    cops = cop_start if cop_start else tuple(sorted(random.sample(nodes, k)))
    rob = robber_start if robber_start else robber_greedy_move(cops, -1, nmap, n)

    # frames: list of (cop_pos, rob_pos, label) for sequential visualization
    frames = []

    # Cops dominate the board from the start — instant win
    if rob == -1:
        print("  Instant capture — cops dominate board from start!")
        frames.append((list(cops), None, "Cops start — instant win"))
        return frames

    frames.append((list(cops), rob, "Initial position"))

    for t in range(max_steps):
        if is_captured(cops, rob, nmap):
            print(f"  Captured at step {t}!")
            break

        # Greedy cop action from V
        best_val = -1e9
        best_cops = cops
        for new_cops_t in get_cop_actions(cops, nmap):
            if is_captured(new_cops_t, rob, nmap):
                val = 1e9
            else:
                rob_next = robber_greedy_move(new_cops_t, rob, nmap, n)
                val = V[(new_cops_t, rob_next)]
            if val > best_val:
                best_val = val
                best_cops = new_cops_t
        cops = best_cops

        # Frame: cops moved, robber hasn't yet
        frames.append((list(cops), rob, f"Turn {t+1}: Cops move"))

        if is_captured(cops, rob, nmap):
            print(f"  Captured at step {t + 1}!")
            break

        rob = robber_greedy_move(cops, rob, nmap, n)
        # Frame: robber responds
        frames.append((list(cops), rob, f"Turn {t+1}: Robber moves"))
    else:
        print(f"  Not captured in {max_steps} steps.")

    return frames


def simulate_mcts_game(mcts_result, cop_start=None, robber_start=None, max_steps=None):
    """
    Simulate a single game using the MCTS policy.
    Returns frames: list of (cop_pos, rob_pos, label) for sequential visualization.
    """
    mcts_action = mcts_result['mcts_action']
    nodes = mcts_result['nodes']
    n = mcts_result['n']
    k = mcts_result['k']
    nmap = mcts_result['neighbors_map']

    if max_steps is None:
        max_steps = max(n * n, 100)

    cops = cop_start if cop_start else tuple(sorted(random.sample(nodes, k)))
    rob = robber_start if robber_start else robber_greedy_move(cops, -1, nmap, n)

    frames = []

    if rob == -1:
        print("  Instant capture — cops dominate board from start!")
        frames.append((list(cops), None, "Cops start — instant win"))
        return frames

    frames.append((list(cops), rob, "Initial position"))

    for t in range(max_steps):
        if is_captured(cops, rob, nmap):
            print(f"  Captured at step {t}!")
            break

        cops = mcts_action(cops, rob)

        frames.append((list(cops), rob, f"Turn {t+1}: Cops move"))

        if is_captured(cops, rob, nmap):
            print(f"  Captured at step {t + 1}!")
            break

        rob = robber_greedy_move(cops, rob, nmap, n)
        frames.append((list(cops), rob, f"Turn {t+1}: Robber moves"))
    else:
        print(f"  Not captured in {max_steps} steps.")

    return frames


def get_state(r_state, c_state, neighbors_map):
    """Build color dict for a single game state."""
    cop_occ = set()
    for cop in c_state:
        cop_occ.update(neighbors_map[cop])
    cop_occ -= set(c_state)

    result = {'blue': set(c_state)}

    if r_state is not None:
        avail = set(available_squares(list(c_state), r_state, neighbors_map))
        result['green'] = avail - {r_state}
        result['red'] = cop_occ - {r_state} - avail
        result['black'] = {r_state}
    else:
        result['red'] = cop_occ

    return result


def display_game(G, frames, neighbors_map, title="Game"):
    """
    Display an interactive game visualization with a slider.
    frames: list of (cop_pos, rob_pos_or_None, label) from simulate_*_game.
    """
    if not frames:
        print(f"{title}: No frames to display.")
        return

    from IPython.display import display as ipy_display
    import ipywidgets as widgets

    n = int(math.sqrt(len(list(G.vertices()))))
    pos = {(i, j): (i, j) for i in range(n) for j in range(n)}
    G.set_pos(pos)
    nx_G = G.networkx_graph()

    states = [get_state(rob, cops, neighbors_map) for cops, rob, label in frames]
    labels = [label for _, _, label in frames]

    def update(turn):
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(nx_G, pos, ax=ax, node_color='lightgrey', node_size=400, with_labels=False)
        for color, node_set in states[turn].items():
            nx.draw_networkx_nodes(nx_G, pos, nodelist=list(node_set),
                                   node_color=color, node_size=400, ax=ax)
        ax.set_title(f"{title} -- {labels[turn]}")
        ax.set_axis_off()
        plt.show()

    slider = widgets.IntSlider(min=0, max=len(states) - 1, step=1, value=0)
    out = widgets.interactive_output(update, {'turn': slider})
    ipy_display(slider, out)


def save_game_gif(G, frames, neighbors_map,
                  gif_path="game.gif", title="Game", dpi=100, duration=500):
    """Save a game as an animated GIF. duration is ms per frame (default 500).
    frames: list of (cop_pos, rob_pos_or_None, label) from simulate_*_game.
    """
    if not frames:
        print(f"{title}: No frames — no GIF to save.")
        return
    import imageio.v2 as imageio
    import os
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend — no popups

    n = int(math.sqrt(len(list(G.vertices()))))
    pos = {(i, j): (i, j) for i in range(n) for j in range(n)}
    G.set_pos(pos)
    nx_G = G.networkx_graph()

    states = [get_state(rob, cops, neighbors_map) for cops, rob, label in frames]
    labels = [label for _, _, label in frames]

    temp_dir = "frames_temp"
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []

    for i, state in enumerate(states):
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(nx_G, pos, ax=ax, node_color='lightgrey', node_size=400, with_labels=False)
        for color, node_set in state.items():
            nx.draw_networkx_nodes(nx_G, pos, nodelist=list(node_set),
                                   node_color=color, node_size=400, ax=ax)
        ax.set_title(f"{title} -- {labels[i]}")
        ax.set_axis_off()
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        fig.savefig(frame_path, dpi=dpi)
        frame_paths.append(frame_path)
        plt.close(fig)

    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=duration)

    for p in frame_paths:
        os.remove(p)
    os.rmdir(temp_dir)
    print(f"GIF saved to {gif_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    sizes = [4, 5]  # nxn queen graphs

    # Value Iteration
    #vi_results = run_vi_sweep(sizes, k=3, gamma=0.99, theta=1e-4, max_iters=200)
    #plot_vi_metrics(vi_results)

    # MCTS (fewer eval episodes for speed -- increase for better estimates)
    #mcts_results = run_mcts_sweep(sizes, k=3, num_simulations=150, num_eval_episodes=50)
    #plot_mcts_metrics(mcts_results)

    # Comparison
    #plot_comparison(vi_results, mcts_results)

    # --- Visualize a single game ---
    # Example: run VI on a 5x5 queen graph, then simulate and save a GIF
    
    n, k = 9, 3

    # vi = value_iteration_cops_robbers(n, k=k)
    # frames = simulate_vi_game(vi)
    G, _, nmap = build_graph_data(n)
    # save_game_gif(G, frames, nmap,
    #               gif_path=f"gifs/vi_{n}x{n}_k{k}.gif", title=f"VI {n}x{n} k={k}")
    # display_game(G, frames, nmap, title=f"VI {n}x{n} k={k}")

    mcts = mcts_cops_robbers(n, k=k, num_simulations=200, num_eval_episodes=1)
    frames = simulate_mcts_game(mcts)
    save_game_gif(G, frames, nmap,
                  gif_path=f"gifs/mcts_{n}x{n}_k{k}.gif", title=f"MCTS {n}x{n} k={k}", duration=750)
    display_game(G, frames, nmap, title=f"MCTS {n}x{n} k={k}")
