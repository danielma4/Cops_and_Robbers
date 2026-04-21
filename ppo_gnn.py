import json
import math
import random
import numpy as np
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_queen_graph(n: int) -> Tuple[List[Tuple[int, int]], Dict, np.ndarray]:
    nodes = [(r, c) for r in range(n) for c in range(n)]
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    neighbors_map = {v: [] for v in nodes}
    edges = []
    for i, (r1, c1) in enumerate(nodes):
        for j, (r2, c2) in enumerate(nodes):
            if i == j:
                continue
            if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                neighbors_map[(r1, c1)].append((r2, c2))
                edges.append((i, j))
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return nodes, neighbors_map, edge_index


def available_squares(cop_pos: List[Tuple], robber_pos: Tuple, neighbors_map: Dict) -> List[Tuple]:
    # cops block their own square and all adjacent squares
    blocked = set(cop_pos)
    for cop in cop_pos:
        blocked.update(neighbors_map[cop])
    
    if robber_pos == (-1, -1) or robber_pos is None:
        return [v for v in neighbors_map if v not in blocked]
    
    moves = set(neighbors_map[robber_pos]) | {robber_pos}
    return [m for m in moves if m not in blocked]


def is_captured(cop_pos: List[Tuple], robber_pos: Tuple, neighbors_map: Dict) -> bool:
    return len(available_squares(cop_pos, robber_pos, neighbors_map)) == 0


def get_SD_length(robber_pos: Tuple, n: int) -> int:
    r, c = robber_pos
    pos_diag = sum(1 for rr in range(n) for cc in range(n) if rr - cc == r - c)
    neg_diag = sum(1 for rr in range(n) for cc in range(n) if rr + cc == r + c)
    return min(pos_diag, neg_diag)


def robber_greedy_move(cop_pos: List[Tuple], robber_pos: Tuple,
                       neighbors_map: Dict, n: int) -> Tuple:
    if robber_pos == (-1, -1) or robber_pos is None:
        avail = available_squares(cop_pos, None, neighbors_map)
    else:
        avail = available_squares(cop_pos, robber_pos, neighbors_map)
    
    if not avail:
        return robber_pos if robber_pos else (-1, -1)
    
    return max(avail, key=lambda m: (
        get_SD_length(m, n),
        len(available_squares(cop_pos, m, neighbors_map))
    ))


class CopsAndRobbersEnv:
    def __init__(self, n: int, k: int = 3, max_steps: int = None):
        self.n = n
        self.k = k
        self.max_steps = max_steps or max(n * n, 100)

        self.nodes, self.neighbors_map, self.edge_index = build_queen_graph(n)
        self.node_to_idx = {v: i for i, v in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)

        self.max_degree = max(len(self.neighbors_map[v]) for v in self.nodes)
        self.M = self.max_degree + 1  # +1 for stay

        self.move_options = {}
        for v in self.nodes:
            self.move_options[v] = [v] + list(self.neighbors_map[v])

        self.cop_pos = None
        self.robber_pos = None
        self.step_count = 0

    def reset(self, cop_start=None, robber_start=None) -> Data:
        if cop_start:
            self.cop_pos = list(cop_start)
        else:
            self.cop_pos = random.sample(self.nodes, self.k)
        
        if robber_start:
            self.robber_pos = robber_start
        else:
            self.robber_pos = robber_greedy_move(self.cop_pos, None, self.neighbors_map, self.n)

        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> Data:
        x = torch.zeros((self.num_nodes, 2), dtype=torch.float32)
        for cop in self.cop_pos:
            x[self.node_to_idx[cop], 0] = 1.0
        if self.robber_pos and self.robber_pos != (-1, -1):
            x[self.node_to_idx[self.robber_pos], 1] = 1.0
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def get_valid_actions(self) -> List[List[int]]:
        return [list(range(len(self.move_options[cop]))) for cop in self.cop_pos]

    def step(self, actions: List[int]) -> Tuple[Data, float, bool, bool, Dict]:
        assert len(actions) == self.k

        new_cop_pos = []
        for i, action in enumerate(actions):
            cop = self.cop_pos[i]
            move_opts = self.move_options[cop]
            new_cop_pos.append(move_opts[action] if action < len(move_opts) else cop)

        self.cop_pos = new_cop_pos
        self.step_count += 1

        if is_captured(self.cop_pos, self.robber_pos, self.neighbors_map):
            return self._get_obs(), 100.0, True, False, {'captured': True, 'steps': self.step_count}

        self.robber_pos = robber_greedy_move(self.cop_pos, self.robber_pos,
                                              self.neighbors_map, self.n)

        if is_captured(self.cop_pos, self.robber_pos, self.neighbors_map):
            return self._get_obs(), 100.0, True, False, {'captured': True, 'steps': self.step_count}

        truncated = self.step_count >= self.max_steps
        avail = len(available_squares(self.cop_pos, self.robber_pos, self.neighbors_map))
        reward = -0.1 - 0.1 * avail
        info = {'captured': False, 'steps': self.step_count, 'robber_avail': avail}
        return self._get_obs(), reward, False, truncated, info


class GNNActorCritic(nn.Module):
    def __init__(self, num_nodes: int, k: int, M: int,
                 hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k
        self.M = M
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(2, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

        self.policy_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, M))
            for _ in range(k)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, data: Data) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, bn in zip(self.convs, self.bn):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = global_mean_pool(x, batch)
        return [head(pooled) for head in self.policy_heads], self.value_head(pooled)

    def get_action_and_value(self, data: Data, valid_actions: List[List[int]] = None,
                             action: List[int] = None):
        pi_logits, value = self.forward(data)

        actions = []
        log_probs = []
        entropies = []

        for i, logits in enumerate(pi_logits):
            if valid_actions is not None:
                mask = torch.full_like(logits, float('-inf'))
                for va in valid_actions[i]:
                    mask[0, va] = 0.0
                logits = logits + mask

            dist = Categorical(logits=logits)
            a = dist.sample() if action is None else torch.tensor([action[i]], device=logits.device)
            actions.append(a.item())
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

        return actions, sum(log_probs), sum(entropies) / self.k, value.squeeze(-1)


class PPOTrainer:
    def __init__(self, env: CopsAndRobbersEnv,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = None):
        self.env = env
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.network = GNNActorCritic(
            num_nodes=env.num_nodes, k=env.k, M=env.M,
            hidden_dim=hidden_dim, num_layers=num_layers
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.metrics = {
            'rewards': [], 'capture_rates': [], 'entropies': [],
            'episode_lengths': [], 'policy_losses': [], 'value_losses': [],
        }
    
    def collect_rollout(self, num_steps: int) -> Dict:
        obs_list, actions_list, log_probs_list = [], [], []
        rewards_list, dones_list, values_list, valid_actions_list = [], [], [], []

        obs = self.env.reset()
        for _ in range(num_steps):
            obs_device = obs.to(self.device)
            valid_actions = self.env.get_valid_actions()
            with torch.no_grad():
                actions, log_prob, _, value = self.network.get_action_and_value(obs_device, valid_actions)
            obs_list.append(obs)
            actions_list.append(actions)
            log_probs_list.append(log_prob.cpu())
            values_list.append(value.cpu())
            valid_actions_list.append(valid_actions)
            obs, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            rewards_list.append(reward)
            dones_list.append(done)
            if done:
                obs = self.env.reset()

        with torch.no_grad():
            _, _, _, final_value = self.network.get_action_and_value(
                obs.to(self.device), self.env.get_valid_actions()
            )

        return {
            'obs': obs_list, 'actions': actions_list,
            'log_probs': torch.stack(log_probs_list),
            'rewards': torch.tensor(rewards_list, dtype=torch.float32),
            'dones': torch.tensor(dones_list, dtype=torch.float32),
            'values': torch.stack(values_list),
            'valid_actions': valid_actions_list,
            'final_value': final_value.cpu(),
        }

    def compute_gae(self, rewards, values, dones, final_value):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        next_value = final_value
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        return advantages, returns

    def update(self, rollout: Dict, num_epochs: int = 4, batch_size: int = 64):
        obs_list = rollout['obs']
        actions_list = rollout['actions']
        old_log_probs = rollout['log_probs'].to(self.device)
        valid_actions_list = rollout['valid_actions']

        advantages, returns = self.compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'], rollout['final_value']
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        num_samples = len(obs_list)
        indices = np.arange(num_samples)
        total_policy_loss = total_value_loss = total_entropy = num_updates = 0

        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start:min(start + batch_size, num_samples)]
                batch_obs = Batch.from_data_list([obs_list[i] for i in batch_indices]).to(self.device)
                batch_actions = [actions_list[i] for i in batch_indices]
                batch_valid = [valid_actions_list[i] for i in batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                pi_logits, values = self.network(batch_obs)
                values = values.squeeze(-1)

                new_log_probs = []
                entropies = []
                for b in range(len(batch_indices)):
                    log_prob_sum = entropy_sum = 0
                    for cop_idx in range(self.env.k):
                        logits = pi_logits[cop_idx][b:b+1]
                        mask = torch.full_like(logits, float('-inf'))
                        for va in batch_valid[b][cop_idx]:
                            mask[0, va] = 0.0
                        logits = logits + mask
                        dist = Categorical(logits=logits)
                        action = torch.tensor([batch_actions[b][cop_idx]], device=self.device)
                        log_prob_sum = log_prob_sum + dist.log_prob(action)
                        entropy_sum = entropy_sum + dist.entropy()
                    new_log_probs.append(log_prob_sum)
                    entropies.append(entropy_sum / self.env.k)

                new_log_probs = torch.cat(new_log_probs)
                entropies = torch.cat(entropies)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropies.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropies.mean().item()
                num_updates += 1

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }

    def evaluate(self, num_episodes: int = 20) -> Dict:
        captures = 0
        episode_lengths = []
        total_rewards = []
        entropies = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            ep_reward = 0
            ep_entropy = []
            for t in range(self.env.max_steps):
                obs_device = obs.to(self.device)
                valid_actions = self.env.get_valid_actions()
                with torch.no_grad():
                    actions, _, entropy, _ = self.network.get_action_and_value(obs_device, valid_actions)
                ep_entropy.append(entropy.item())
                obs, reward, terminated, truncated, info = self.env.step(actions)
                ep_reward += reward
                if terminated:
                    captures += 1
                    episode_lengths.append(t + 1)
                    break
                if truncated:
                    episode_lengths.append(t + 1)
                    break
            else:
                episode_lengths.append(self.env.max_steps)
            total_rewards.append(ep_reward)
            entropies.append(np.mean(ep_entropy) if ep_entropy else 0)

        return {
            'capture_rate': captures / num_episodes,
            'avg_episode_length': np.mean(episode_lengths),
            'avg_reward': np.mean(total_rewards),
            'avg_entropy': np.mean(entropies),
        }

    def train(self, num_iterations: int = 100, steps_per_iter: int = 2048,
              eval_interval: int = 10, eval_episodes: int = 20,
              save_path: str = None):
        print(f"training ppo on {self.env.n}x{self.env.n}, k={self.env.k}, device={self.device}")

        for iteration in tqdm(range(1, num_iterations + 1), desc="PPO"):
            rollout = self.collect_rollout(steps_per_iter)
            update_info = self.update(rollout)
            self.metrics['policy_losses'].append(update_info['policy_loss'])
            self.metrics['value_losses'].append(update_info['value_loss'])

            if iteration % eval_interval == 0:
                eval_info = self.evaluate(eval_episodes)
                self.metrics['rewards'].append(eval_info['avg_reward'])
                self.metrics['capture_rates'].append(eval_info['capture_rate'])
                self.metrics['entropies'].append(eval_info['avg_entropy'])
                self.metrics['episode_lengths'].append(eval_info['avg_episode_length'])
                print(f"\niter {iteration}: capture={eval_info['capture_rate']:.2%}, "
                      f"ep_len={eval_info['avg_episode_length']:.1f}, "
                      f"reward={eval_info['avg_reward']:.1f}")
                if save_path:
                    self.save_metrics(save_path)

        final_eval = self.evaluate(eval_episodes * 2)
        print(f"\nfinal: capture={final_eval['capture_rate']:.2%}, "
              f"ep_len={final_eval['avg_episode_length']:.1f}")
        return self.metrics

    def save_metrics(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n': self.env.n, 'k': self.env.k, 'M': self.env.M,
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class GNNDQN(nn.Module):
    def __init__(self, num_nodes: int, k: int, M: int,
                 hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.k = k
        self.M = M

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(2, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        self.q_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, M))
            for _ in range(k)
        ])

    def forward(self, data: Data) -> List[torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv, bn in zip(self.convs, self.bn):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        pooled = global_mean_pool(x, batch)
        return [head(pooled) for head in self.q_heads]


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, actions, reward, next_obs, done, valid_actions, next_valid_actions):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, actions, reward, next_obs, done,
                                      valid_actions, next_valid_actions)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, valid_actions, next_valid_actions = zip(*batch)
        return (list(obs), list(actions),
                torch.tensor(rewards, dtype=torch.float32),
                list(next_obs),
                torch.tensor(dones, dtype=torch.float32),
                list(valid_actions), list(next_valid_actions))

    
    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, env: CopsAndRobbersEnv,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 10000,
                 target_update: int = 1000,
                 buffer_size: int = 100000,
                 device: str = None):
        self.env = env
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.q_network = GNNDQN(
            num_nodes=env.num_nodes, k=env.k, M=env.M,
            hidden_dim=hidden_dim, num_layers=num_layers
        ).to(self.device)
        self.target_network = GNNDQN(
            num_nodes=env.num_nodes, k=env.k, M=env.M,
            hidden_dim=hidden_dim, num_layers=num_layers
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.metrics = {
            'rewards': [], 'capture_rates': [], 'entropies': [],
            'episode_lengths': [], 'q_losses': [],
        }
        self.steps_done = 0
        self.best_capture_rate = 0.0
        self.best_model_state = None
    
    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, obs, valid_actions, epsilon: float = None):
        if epsilon is None:
            epsilon = self.get_epsilon()
        if random.random() < epsilon:
            return [random.choice(va) for va in valid_actions]
        obs_device = obs.to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_device)
        actions = []
        for i, q_vals in enumerate(q_values):
            q_masked = q_vals.clone()
            mask = torch.full_like(q_masked, float('-inf'))
            for va in valid_actions[i]:
                mask[0, va] = 0.0
            q_masked = q_masked + mask
            actions.append(q_masked.argmax(dim=1).item())
        return actions
    
    def update(self, batch_size: int = 64):
        if len(self.replay_buffer) < batch_size:
            return {'q_loss': 0.0}
        (obs_list, actions_list, rewards, next_obs_list, dones,
         valid_actions_list, next_valid_actions_list) = self.replay_buffer.sample(batch_size)
        batch_obs = Batch.from_data_list(obs_list).to(self.device)
        batch_next_obs = Batch.from_data_list(next_obs_list).to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_network(batch_obs)
        current_q = []
        for i in range(self.env.k):
            actions_i = torch.tensor([a[i] for a in actions_list],
                                     dtype=torch.long, device=self.device)
            q_i = q_values[i].gather(1, actions_i.unsqueeze(1)).squeeze(1)
            current_q.append(q_i)
        current_q = torch.stack(current_q).mean(dim=0)

        with torch.no_grad():
            next_q_values = self.target_network(batch_next_obs)
            next_q = []
            for i in range(self.env.k):
                q_i = next_q_values[i]
                for b in range(len(next_valid_actions_list)):
                    mask = torch.full((1, self.env.M), float('-inf'), device=self.device)
                    for va in next_valid_actions_list[b][i]:
                        mask[0, va] = 0.0
                    q_i[b:b+1] = q_i[b:b+1] + mask
                next_q.append(q_i.max(dim=1)[0])
            next_q = torch.stack(next_q).mean(dim=0)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        return {'q_loss': loss.item()}
    
    def evaluate(self, num_episodes: int = 20, epsilon: float = 0.0) -> Dict:
        self.q_network.eval()
        captures = 0
        episode_lengths = []
        total_rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            ep_reward = 0
            for t in range(self.env.max_steps):
                valid_actions = self.env.get_valid_actions()
                actions = self.select_action(obs, valid_actions, epsilon=epsilon)
                obs, reward, terminated, truncated, info = self.env.step(actions)
                ep_reward += reward
                if terminated:
                    captures += 1
                    episode_lengths.append(t + 1)
                    break
                if truncated:
                    episode_lengths.append(t + 1)
                    break
            else:
                episode_lengths.append(self.env.max_steps)
            total_rewards.append(ep_reward)
        self.q_network.train()
        return {
            'capture_rate': captures / num_episodes,
            'avg_episode_length': np.mean(episode_lengths),
            'avg_reward': np.mean(total_rewards),
        }
    
    def train(self, num_steps: int = 200000, batch_size: int = 64,
              eval_interval: int = 10000, eval_episodes: int = 20,
              save_path: str = None):
        print(f"training dqn on {self.env.n}x{self.env.n}, k={self.env.k}, device={self.device}")
        obs = self.env.reset()
        episode_reward = episode_length = 0
        pbar = tqdm(range(num_steps), desc="DQN")
        for step in pbar:
            valid_actions = self.env.get_valid_actions()
            actions = self.select_action(obs, valid_actions)
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            next_valid_actions = self.env.get_valid_actions()
            obs_cpu = obs.cpu() if hasattr(obs, 'cpu') else obs
            next_obs_cpu = next_obs.cpu() if hasattr(next_obs, 'cpu') else next_obs
            self.replay_buffer.push(obs_cpu, actions, reward, next_obs_cpu, done,
                                    valid_actions, next_valid_actions)
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.steps_done += 1
            if len(self.replay_buffer) >= batch_size:
                update_info = self.update(batch_size)
                self.metrics['q_losses'].append(update_info['q_loss'])
            if self.steps_done % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            if done:
                obs = self.env.reset()
                episode_reward = episode_length = 0
            if (step + 1) % eval_interval == 0:
                eval_info = self.evaluate(eval_episodes)
                self.metrics['rewards'].append(eval_info['avg_reward'])
                self.metrics['capture_rates'].append(eval_info['capture_rate'])
                self.metrics['episode_lengths'].append(eval_info['avg_episode_length'])
                self.metrics['entropies'].append(0)
                epsilon = self.get_epsilon()
                if eval_info['capture_rate'] > self.best_capture_rate:
                    self.best_capture_rate = eval_info['capture_rate']
                    self.best_model_state = {
                        'q_network_state_dict': self.q_network.state_dict(),
                        'target_network_state_dict': self.target_network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'steps_done': self.steps_done,
                        'capture_rate': self.best_capture_rate,
                        'n': self.env.n, 'k': self.env.k, 'M': self.env.M,
                    }
                pbar.set_postfix({
                    'capture': f"{eval_info['capture_rate']:.2%}",
                    'ep_len': f"{eval_info['avg_episode_length']:.1f}",
                    'epsilon': f"{epsilon:.3f}",
                    'best': f"{self.best_capture_rate:.2%}"
                })
                if save_path:
                    self.save_metrics(save_path)
        return self.metrics

    def save_metrics(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def save_model(self, path: str, save_best: bool = False):
        if save_best and self.best_model_state is not None:
            torch.save(self.best_model_state, path)
        else:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'steps_done': self.steps_done,
                'n': self.env.n, 'k': self.env.k, 'M': self.env.M,
            }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)


def train_dqn(n: int = 7, k: int = 3, num_steps: int = 200000,
              batch_size: int = 64, hidden_dim: int = 128,
              lr: float = 1e-3, save_dir: str = './dqn_checkpoints'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    env = CopsAndRobbersEnv(n=n, k=k)
    trainer = DQNTrainer(env, hidden_dim=hidden_dim, lr=lr)
    metrics_path = os.path.join(save_dir, f'dqn_metrics_n{n}_k{k}.json')
    metrics = trainer.train(
        num_steps=num_steps, batch_size=batch_size,
        eval_interval=10000, eval_episodes=20, save_path=metrics_path
    )
    model_path = os.path.join(save_dir, f'dqn_model_n{n}_k{k}.pt')
    trainer.save_model(model_path)
    best_model_path = os.path.join(save_dir, f'dqn_model_best_n{n}_k{k}.pt')
    trainer.save_model(best_model_path, save_best=True)
    plot_path = os.path.join(save_dir, f'dqn_training_metrics_n{n}_k{k}.png')
    plot_training_metrics(metrics, plot_path)
    return trainer, metrics


def plot_training_metrics(metrics: Dict, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    if metrics['rewards']:
        ax.plot(metrics['rewards'], 'b-', linewidth=2)
    ax.set_xlabel('Evaluation'); ax.set_ylabel('Average Reward')
    ax.set_title('Training Reward'); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    if metrics['capture_rates']:
        ax.plot(metrics['capture_rates'], 'g-', linewidth=2)
    ax.set_xlabel('Evaluation'); ax.set_ylabel('Capture Rate')
    ax.set_title('Capture Rate'); ax.set_ylim([-0.1, 1.1]); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    if metrics['entropies']:
        ax.plot(metrics['entropies'], 'r-', linewidth=2)
    ax.set_xlabel('Evaluation'); ax.set_ylabel('Policy Entropy')
    ax.set_title('Policy Entropy'); ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    if metrics['episode_lengths']:
        ax.plot(metrics['episode_lengths'], 'm-', linewidth=2)
    ax.set_xlabel('Evaluation'); ax.set_ylabel('Episode Length')
    ax.set_title('Average Episode Length'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def train_ppo(n: int = 7, k: int = 3, num_iterations: int = 100,
              steps_per_iter: int = 2048, hidden_dim: int = 128,
              lr: float = 3e-4, save_dir: str = './ppo_checkpoints'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    env = CopsAndRobbersEnv(n=n, k=k)
    trainer = PPOTrainer(env, hidden_dim=hidden_dim, lr=lr)
    metrics_path = os.path.join(save_dir, f'metrics_n{n}_k{k}.json')
    metrics = trainer.train(
        num_iterations=num_iterations, steps_per_iter=steps_per_iter,
        eval_interval=10, eval_episodes=20, save_path=metrics_path
    )
    model_path = os.path.join(save_dir, f'model_n{n}_k{k}.pt')
    trainer.save_model(model_path)
    plot_path = os.path.join(save_dir, f'training_metrics_n{n}_k{k}.png')
    plot_training_metrics(metrics, plot_path)
    return trainer, metrics


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ppo', choices=['ppo', 'dqn'])
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    if args.method == 'dqn':
        if args.resume:
            env = CopsAndRobbersEnv(n=args.n, k=args.k)
            trainer = DQNTrainer(env, hidden_dim=args.hidden, lr=args.lr)
            trainer.load_model(args.resume)
            metrics_path = os.path.join(args.save_dir, f'dqn_metrics_n{args.n}_k{args.k}.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    trainer.metrics = json.load(f)
            metrics = trainer.train(
                num_steps=args.steps, batch_size=args.batch_size,
                eval_interval=10000, eval_episodes=20, save_path=metrics_path
            )
            model_path = os.path.join(args.save_dir, f'dqn_model_n{args.n}_k{args.k}.pt')
            trainer.save_model(model_path)
            plot_path = os.path.join(args.save_dir, f'dqn_training_metrics_n{args.n}_k{args.k}.png')
            plot_training_metrics(metrics, plot_path)
        else:
            train_dqn(
                n=args.n, k=args.k,
                num_steps=args.steps if args.steps > 1000 else 200000,
                batch_size=args.batch_size, hidden_dim=args.hidden,
                lr=args.lr, save_dir=args.save_dir
            )
    else:
        if args.resume:
            env = CopsAndRobbersEnv(n=args.n, k=args.k)
            trainer = PPOTrainer(env, hidden_dim=args.hidden, lr=args.lr)
            trainer.load_model(args.resume)
            metrics_path = os.path.join(args.save_dir, f'metrics_n{args.n}_k{args.k}.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    trainer.metrics = json.load(f)
            metrics = trainer.train(
                num_iterations=args.iterations, steps_per_iter=args.steps,
                eval_interval=10, eval_episodes=20, save_path=metrics_path
            )
            model_path = os.path.join(args.save_dir, f'model_n{args.n}_k{args.k}.pt')
            trainer.save_model(model_path)
            plot_path = os.path.join(args.save_dir, f'training_metrics_n{args.n}_k{args.k}.png')
            plot_training_metrics(metrics, plot_path)
        else:
            train_ppo(
                n=args.n, k=args.k, num_iterations=args.iterations,
                steps_per_iter=args.steps, hidden_dim=args.hidden,
                lr=args.lr, save_dir=args.save_dir
            )
