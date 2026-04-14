"""
GNN architecture for Cops and Robbers AlphaZero.

Input:  node features (n^2, 2) — [cop_presence, robber_presence] per node
        + edge_index (2, E) for the queen graph
Output: policy logits (action_size,) + value scalar in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class CopsAndRobbersGNN(nn.Module):

    def __init__(self, num_nodes, action_size, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.action_size = action_size
        self.dropout = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(2, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Policy head: global pool → MLP → action logits
        self.policy_fc1 = nn.Linear(hidden_dim, 256)
        self.policy_fc2 = nn.Linear(256, action_size)

        # Value head: global pool → MLP → scalar
        self.value_fc1 = nn.Linear(hidden_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x, edge_index, batch=None):
        """
        x:          (total_nodes_in_batch, 2)
        edge_index: (2, total_edges_in_batch)
        batch:      (total_nodes_in_batch,) — graph membership for each node
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, bn in zip(self.convs, self.bn):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling → one vector per graph in batch
        pooled = global_mean_pool(x, batch)  # (batch_size, hidden_dim)

        # Policy head
        pi = F.relu(self.policy_fc1(pooled))
        pi = F.dropout(pi, p=self.dropout, training=self.training)
        pi = self.policy_fc2(pi)
        pi = F.log_softmax(pi, dim=1)

        # Value head
        v = F.relu(self.value_fc1(pooled))
        v = F.dropout(v, p=self.dropout, training=self.training)
        v = self.value_fc2(v)
        v = torch.tanh(v)

        return pi, v
