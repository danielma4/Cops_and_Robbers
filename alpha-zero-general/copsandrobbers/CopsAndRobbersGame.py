"""
Cops and Robbers game for alpha-zero-general.

Two-player model:
  Player  1 = cops (k cops, joint action each turn)
  Player -1 = robber (deterministic greedy — always 1 valid move)

Board representation (compact numpy array):
  board[0:k]   = sorted cop node indices
  board[k]     = robber node index (-1 if instant capture)
  board[k+1]   = step count
  board[k+2]   = phase (0 = cop turn, 1 = robber turn)

Action space = M^k  where M = max_node_degree + 1.
  Each cop picks a local action index (0 = stay, 1..deg = move to i-th neighbor).
  Joint action = a0*M^(k-1) + a1*M^(k-2) + ... + a_{k-1}.
  On robber turns, only 1 action is valid (the greedy move).
"""

import sys
import numpy as np
import random

sys.path.append('..')
from Game import Game


class CopsAndRobbersGame(Game):

    def __init__(self, n, k=3, max_steps=None):
        self.n = n
        self.k = k
        self.n2 = n * n
        self.max_steps = max_steps or max(n * n, 100)
        self._build_graph()
        self.M = self.max_degree + 1          # per-cop branching factor
        self._action_size = self.M ** k       # joint cop action space (also used for robber)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        nodes = [(r, c) for r in range(self.n) for c in range(self.n)]
        self.idx_to_node = nodes
        self.node_to_idx = {v: i for i, v in enumerate(nodes)}

        # Adjacency lists (by node index)
        self.neighbors = [[] for _ in range(self.n2)]
        for i, (r1, c1) in enumerate(nodes):
            for j, (r2, c2) in enumerate(nodes):
                if i != j and (r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2)):
                    self.neighbors[i].append(j)

        self.max_degree = max(len(nb) for nb in self.neighbors)

        # Per-node move options: [stay] + sorted(neighbors)
        self.move_options = [
            [i] + sorted(self.neighbors[i]) for i in range(self.n2)
        ]

        # Edge index for PyG (2 x E)
        rows, cols = [], []
        for i, nbrs in enumerate(self.neighbors):
            for j in nbrs:
                rows.append(i)
                cols.append(j)
        self.edge_index = np.array([rows, cols], dtype=np.int64)

        # Precompute SD length for each node
        self.sd_length = np.zeros(self.n2, dtype=int)
        for idx, (r, c) in enumerate(nodes):
            pos_len = sum(1 for rr in range(self.n) for cc in range(self.n) if rr - cc == r - c)
            neg_len = sum(1 for rr in range(self.n) for cc in range(self.n) if rr + cc == r + c)
            self.sd_length[idx] = min(pos_len, neg_len)

    # ------------------------------------------------------------------
    # Game logic helpers
    # ------------------------------------------------------------------

    def _available_squares(self, cop_nodes, rob_node):
        """Squares available to robber. Cops dominate positions + neighbors."""
        blocked = set(cop_nodes)
        for c in cop_nodes:
            blocked.update(self.neighbors[c])
        if rob_node == -1:  # initial placement
            return [v for v in range(self.n2) if v not in blocked]
        moves = set(self.neighbors[rob_node]) | {rob_node}
        return [m for m in moves if m not in blocked]

    def _is_captured(self, cop_nodes, rob_node):
        return len(self._available_squares(cop_nodes, rob_node)) == 0

    def _robber_greedy_move(self, cop_nodes, rob_node):
        """Robber picks move maximizing (SD length, # available squares)."""
        avail = self._available_squares(cop_nodes, rob_node)
        if not avail:
            return rob_node  # already captured
        return max(avail, key=lambda m: (
            self.sd_length[m],
            len(self._available_squares(cop_nodes, m))
        ))

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def _encode_action(self, local_actions):
        a = 0
        for i, la in enumerate(local_actions):
            a += la * (self.M ** (self.k - 1 - i))
        return a

    def _decode_action(self, action):
        local_actions = []
        rem = action
        for i in range(self.k):
            divisor = self.M ** (self.k - 1 - i)
            la = rem // divisor
            rem %= divisor
            local_actions.append(la)
        return local_actions

    def _action_to_targets(self, cop_nodes, action):
        """Decode joint action to target node indices for each cop."""
        local_actions = self._decode_action(action)
        targets = []
        for i, la in enumerate(local_actions):
            opts = self.move_options[cop_nodes[i]]
            if la < len(opts):
                targets.append(opts[la])
            else:
                targets.append(cop_nodes[i])  # fallback: stay
        return targets

    def _robber_action_encode(self, cop_nodes, rob_node):
        """Encode the robber's greedy move as an action index in [0, action_size).
        We map it to the first valid index (pad with zeros for other cops)."""
        target = self._robber_greedy_move(cop_nodes, rob_node)
        # Encode as: cop0 stays (0), cop1 stays (0), ..., and we use the
        # target node's index. But action space is M^k for joint cop moves.
        # For robber turns, we just use action index = target node index.
        # Since target < n^2 < M^k, this fits in the action space.
        return target

    # ------------------------------------------------------------------
    # Board packing / unpacking
    # ------------------------------------------------------------------

    def _pack_board(self, cop_nodes, rob_node, step, phase):
        board = np.zeros(self.k + 3, dtype=np.float64)
        board[:self.k] = sorted(cop_nodes)
        board[self.k] = rob_node
        board[self.k + 1] = step
        board[self.k + 2] = phase
        return board

    def _unpack_board(self, board):
        cop_nodes = list(board[:self.k].astype(int))
        rob_node = int(board[self.k])
        step = int(board[self.k + 1])
        phase = int(board[self.k + 2])
        return cop_nodes, rob_node, step, phase

    # ------------------------------------------------------------------
    # Game interface (required by alpha-zero-general)
    # ------------------------------------------------------------------

    def getInitBoard(self):
        cop_nodes = sorted(random.sample(range(self.n2), self.k))
        rob_node = self._robber_greedy_move(cop_nodes, -1)  # greedy start
        if rob_node == -1:
            rob_node = -1  # instant capture (rare but possible)
        return self._pack_board(cop_nodes, rob_node, 0, 0)

    def getBoardSize(self):
        return (self.n2, 2)  # node features shape for GNN

    def getActionSize(self):
        return self._action_size

    def getNextState(self, board, player, action):
        cop_nodes, rob_node, step, phase = self._unpack_board(board)

        if phase == 0:
            # Cop turn (player 1): apply joint cop action
            targets = self._action_to_targets(cop_nodes, action)
            new_cops = sorted(targets)
            new_board = self._pack_board(new_cops, rob_node, step, 1)
            return new_board, -1  # robber's turn next

        else:
            # Robber turn (player -1): apply greedy move
            # action argument is ignored; we always use greedy
            if rob_node >= 0 and not self._is_captured(cop_nodes, rob_node):
                new_rob = self._robber_greedy_move(cop_nodes, rob_node)
            else:
                new_rob = rob_node
            new_board = self._pack_board(cop_nodes, new_rob, step + 1, 0)
            return new_board, 1  # cop's turn next

    def getValidMoves(self, board, player):
        cop_nodes, rob_node, step, phase = self._unpack_board(board)
        valids = np.zeros(self._action_size, dtype=np.int8)

        if phase == 0:
            # Cop turn: valid if all local actions are in range
            for a in range(self._action_size):
                local_actions = self._decode_action(a)
                valid = True
                for i, la in enumerate(local_actions):
                    if la >= len(self.move_options[cop_nodes[i]]):
                        valid = False
                        break
                if valid:
                    valids[a] = 1
        else:
            # Robber turn: single valid action (greedy move)
            rob_target = self._robber_action_encode(cop_nodes, rob_node)
            if 0 <= rob_target < self._action_size:
                valids[rob_target] = 1
            else:
                # Fallback: robber stays (captured state, action 0)
                valids[0] = 1

        return valids

    def getGameEnded(self, board, player):
        cop_nodes, rob_node, step, phase = self._unpack_board(board)

        # Instant capture (robber had nowhere to start)
        if rob_node == -1:
            return 1 if player == 1 else -1

        # After cops moved (phase 1): check capture
        if phase == 1 and self._is_captured(cop_nodes, rob_node):
            # Cops win
            return 1 if player == -1 else -1
            # Note: at phase 1, player is -1 (robber).
            # Cops winning is bad for robber → return -1 from robber's perspective.
            # But the convention: return +1 if `player` won.
            # Robber did NOT win, so return -1.

        # Timeout
        if step >= self.max_steps:
            # Cops lose (failed to capture)
            return -1 if player == 1 else 1

        return 0

    def getCanonicalForm(self, board, player):
        # Board already contains all info. No transformation needed.
        return board

    def getSymmetries(self, board, pi):
        # No symmetry augmentation for now (could add board rotations later)
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def display(self, board):
        cop_nodes, rob_node, step, phase = self._unpack_board(board)
        phase_str = "Cops" if phase == 0 else "Robber"
        cop_pos = [self.idx_to_node[c] for c in cop_nodes]
        rob_pos = self.idx_to_node[rob_node] if rob_node >= 0 else "N/A"
        print(f"Step {step} | {phase_str}'s turn | Cops: {cop_pos} | Robber: {rob_pos}")
