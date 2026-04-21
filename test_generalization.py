import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List

import torch

from ppo_gnn import (
    CopsAndRobbersEnv, GNNActorCritic, GNNDQN,
    build_queen_graph, is_captured, robber_greedy_move,
    plot_training_metrics
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate_on_size(network, n: int, k: int = 3,
                     num_episodes: int = 100, use_mcts: bool = False,
                     mcts_sims: int = 50, device: str = 'cpu') -> Dict:
    # evaluate a trained network on an nxn environment, adapting M if needed
    env = CopsAndRobbersEnv(n=n, k=k)
    is_dqn = isinstance(network, GNNDQN)
    train_M = network.M
    eval_M = env.M

    if train_M != eval_M:
        if is_dqn:
            eval_network = GNNDQN(
                num_nodes=env.num_nodes, k=k, M=eval_M,
                hidden_dim=network.hidden_dim, num_layers=len(network.convs)
            ).to(device)
        else:
            eval_network = GNNActorCritic(
                num_nodes=env.num_nodes, k=k, M=eval_M,
                hidden_dim=network.hidden_dim, num_layers=len(network.convs)
            ).to(device)
        for src_conv, dst_conv in zip(network.convs, eval_network.convs):
            dst_conv.load_state_dict(src_conv.state_dict())
        for src_bn, dst_bn in zip(network.bn, eval_network.bn):
            dst_bn.load_state_dict(src_bn.state_dict())
        network = eval_network

    network.eval()
    if use_mcts:
        use_mcts = False

    captures = 0
    episode_lengths = []
    total_rewards = []
    entropies = []

    for _ in tqdm(range(num_episodes), desc=f"  eval {n}x{n}", leave=False):
        obs = env.reset()
        ep_reward = 0
        ep_entropy = []
        for t in range(env.max_steps):
            obs_device = obs.to(device)
            valid_actions = env.get_valid_actions()
            with torch.no_grad():
                if is_dqn:
                    q_values = network(obs_device)
                    actions = []
                    for i, q_vals in enumerate(q_values):
                        q_masked = q_vals.clone()
                        mask = torch.full_like(q_masked, float('-inf'))
                        for va in valid_actions[i]:
                            mask[0, va] = 0.0
                        q_masked = q_masked + mask
                        actions.append(q_masked.argmax(dim=1).item())
                    ep_entropy.append(0)
                else:
                    actions, _, entropy, _ = network.get_action_and_value(obs_device, valid_actions)
                    ep_entropy.append(entropy.item())
            obs, reward, terminated, truncated, info = env.step(actions)
            ep_reward += reward
            if terminated:
                captures += 1
                episode_lengths.append(t + 1)
                break
            if truncated:
                episode_lengths.append(t + 1)
                break
        else:
            episode_lengths.append(env.max_steps)
        total_rewards.append(ep_reward)
        entropies.append(np.mean(ep_entropy) if ep_entropy else 0)

    return {
        'n': n,
        'capture_rate': captures / num_episodes,
        'avg_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'avg_reward': np.mean(total_rewards),
        'avg_entropy': np.mean(entropies),
    }


def test_generalization(model_path: str, train_n: int, eval_sizes: List[int],
                        k: int = 3, num_episodes: int = 100,
                        use_mcts: bool = False, mcts_sims: int = 50,
                        save_path: str = None) -> List[Dict]:
    # load a checkpoint and evaluate generalization across multiple graph sizes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    train_env = CopsAndRobbersEnv(n=train_n, k=k)

    if 'q_network_state_dict' in checkpoint:
        network = GNNDQN(
            num_nodes=train_env.num_nodes, k=k, M=train_env.M,
            hidden_dim=128, num_layers=3
        ).to(device)
        network.load_state_dict(checkpoint['q_network_state_dict'])
    else:
        network = GNNActorCritic(
            num_nodes=train_env.num_nodes, k=k, M=train_env.M,
            hidden_dim=128, num_layers=3
        ).to(device)
        network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    results = []
    for n in eval_sizes:
        result = evaluate_on_size(
            network, n, k=k, num_episodes=num_episodes,
            use_mcts=use_mcts, mcts_sims=mcts_sims, device=device
        )
        results.append(result)
        print(f"{n}x{n}: capture={result['capture_rate']:.2%}, "
              f"ep_len={result['avg_episode_length']:.1f}±{result['std_episode_length']:.1f}")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump({
                'train_n': train_n, 'k': k, 'use_mcts': use_mcts,
                'mcts_sims': mcts_sims, 'num_episodes': num_episodes, 'results': results
            }, f, indent=2)
    return results


def plot_generalization_results(results: List[Dict], train_n: int,
                                save_path: str = None):
    # plot capture rate, episode length, reward, and entropy across eval sizes
    sizes = [r['n'] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Generalization Results (Trained on {train_n}x{train_n})', fontsize=14)
    capture_rates = [r['capture_rate'] for r in results]
    colors = ['green' if s == train_n else 'steelblue' for s in sizes]
    ax = axes[0, 0]
    ax.bar(range(len(sizes)), capture_rates, color=colors)
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.set_ylabel('Capture Rate'); ax.set_title('Capture Rate vs Graph Size'); ax.set_ylim(0, 1.05)
    if train_n in sizes:
        ax.axhline(y=capture_rates[sizes.index(train_n)], color='green', linestyle='--', alpha=0.5)
    for i, cr in enumerate(capture_rates):
        ax.text(i, cr + 0.02, f'{cr:.0%}', ha='center', fontsize=9)
    ep_lens = [r['avg_episode_length'] for r in results]
    ep_stds = [r['std_episode_length'] for r in results]
    ax = axes[0, 1]
    ax.bar(range(len(sizes)), ep_lens, yerr=ep_stds, color=colors, capsize=3)
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.set_ylabel('Avg Episode Length'); ax.set_title('Episode Length vs Graph Size')
    ax = axes[1, 0]
    ax.bar(range(len(sizes)), [r['avg_reward'] for r in results], color=colors)
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.set_ylabel('Avg Reward'); ax.set_title('Average Reward vs Graph Size')
    ax = axes[1, 1]
    ax.bar(range(len(sizes)), [r['avg_entropy'] for r in results], color=colors)
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.set_ylabel('Avg Entropy'); ax.set_title('Policy Entropy vs Graph Size')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def run_baseline_comparison(eval_sizes: List[int], k: int = 3,
                            num_episodes: int = 50) -> List[Dict]:
    # evaluate random cop baseline across multiple graph sizes
    results = []
    for n in eval_sizes:
        env = CopsAndRobbersEnv(n=n, k=k)
        captures = 0
        episode_lengths = []
        for _ in tqdm(range(num_episodes), desc=f"  random {n}x{n}", leave=False):
            env.reset()
            for t in range(env.max_steps):
                valid = env.get_valid_actions()
                actions = [np.random.choice(va) for va in valid]
                _, _, terminated, truncated, _ = env.step(actions)
                if terminated:
                    captures += 1
                    episode_lengths.append(t + 1)
                    break
                if truncated:
                    episode_lengths.append(t + 1)
                    break
            else:
                episode_lengths.append(env.max_steps)
        results.append({
            'n': n,
            'capture_rate': captures / num_episodes,
            'avg_episode_length': np.mean(episode_lengths),
        })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--train_n', type=int, required=True)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--eval_sizes', type=int, nargs='+', default=[4, 5, 6, 7, 8, 9])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--mcts', action='store_true')
    parser.add_argument('--mcts_sims', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./generalization_results')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    import os
    os.makedirs(args.save_dir, exist_ok=True)

    results_path = os.path.join(args.save_dir, f'results_train{args.train_n}.json')
    results = test_generalization(
        model_path=args.model, train_n=args.train_n, eval_sizes=args.eval_sizes,
        k=args.k, num_episodes=args.episodes, use_mcts=args.mcts,
        mcts_sims=args.mcts_sims, save_path=results_path
    )
    plot_path = os.path.join(args.save_dir, f'generalization_train{args.train_n}.png')
    plot_generalization_results(results, args.train_n, plot_path)
    if args.baseline:
        baseline_results = run_baseline_comparison(args.eval_sizes, args.k, args.episodes // 2)
        baseline_path = os.path.join(args.save_dir, 'baseline_random.json')
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
