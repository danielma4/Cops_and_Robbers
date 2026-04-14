"""
NeuralNet wrapper for the Cops and Robbers GNN.

Handles conversion between the compact board representation
(used by the Game class) and PyG graph data (used by the GNN).
"""

import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch

from .CopsAndRobbersNNet import CopsAndRobbersGNN

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'hidden_dim': 128,
    'num_layers': 3,
})


class NNetWrapper(NeuralNet):

    def __init__(self, game):
        self.game = game
        self.n2 = game.n2
        self.k = game.k
        self.action_size = game.getActionSize()

        self.nnet = CopsAndRobbersGNN(
            num_nodes=self.n2,
            action_size=self.action_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

        # Fixed edge_index for the queen graph (same for all states)
        self.edge_index = torch.LongTensor(game.edge_index)

        if args.cuda:
            self.nnet.cuda()
            self.edge_index = self.edge_index.cuda()

    def _board_to_node_features(self, board):
        """Convert compact board array to node feature matrix (n^2, 2)."""
        x = np.zeros((self.n2, 2), dtype=np.float32)
        cop_nodes = board[:self.k].astype(int)
        rob_node = int(board[self.k])
        for c in cop_nodes:
            if 0 <= c < self.n2:
                x[c, 0] += 1  # multiple cops can share a node
        if 0 <= rob_node < self.n2:
            x[rob_node, 1] = 1
        return x

    def _boards_to_batch(self, boards):
        """Convert list of compact boards to a PyG Batch."""
        data_list = []
        for board in boards:
            x = torch.FloatTensor(self._board_to_node_features(board))
            data = Data(x=x, edge_index=self.edge_index.cpu())
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        if args.cuda:
            batch = batch.cuda()
        return batch

    def train(self, examples):
        """
        examples: list of (board, pi, v) tuples
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print(f'EPOCH ::: {epoch + 1}')
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert boards to PyG batch
                batch = self._boards_to_batch(boards)

                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if args.cuda:
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # Forward pass
                out_pi, out_v = self.nnet(batch.x, batch.edge_index, batch.batch)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), len(boards))
                v_losses.update(l_v.item(), len(boards))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: compact numpy array
        Returns: (pi, v) — policy probabilities and value scalar
        """
        x = torch.FloatTensor(self._board_to_node_features(board))
        edge_index = self.edge_index

        if args.cuda:
            x = x.cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(x, edge_index, batch=None)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
