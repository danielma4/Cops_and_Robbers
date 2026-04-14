"""
AlphaZero training for Cops and Robbers on Queen Graphs.

Usage:
    python main_cr.py

Adjust n (board size), k (number of cops), and training hyperparameters below.
"""

import logging
import sys

from utils import dotdict
from Coach import Coach
from copsandrobbers.CopsAndRobbersGame import CopsAndRobbersGame
from copsandrobbers.pytorch.NNet import NNetWrapper as NNet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---- Game configuration ----
n = 4       # board size (n x n queen graph)
k = 3       # number of cops

# ---- Training hyperparameters ----
args = dotdict({
    'numIters': 50,                    # number of training iterations
    'numEps': 25,                      # self-play episodes per iteration
    'tempThreshold': 10,               # steps before switching to greedy (temp=0)
    'updateThreshold': 0.55,           # win rate to accept new model
    'maxlenOfQueue': 100000,           # max training examples in buffer
    'numMCTSSims': 25,                 # MCTS simulations per move
    'arenaCompare': 20,                # games to compare old vs new model
    'cpuct': 1.0,                      # UCB exploration constant
    'checkpoint': f'./temp_cr_{n}x{n}_k{k}/',
    'load_model': False,
    'load_folder_file': (f'./temp_cr_{n}x{n}_k{k}/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info(f'Setting up Cops and Robbers: {n}x{n} queen graph, k={k} cops')

    game = CopsAndRobbersGame(n, k=k)
    log.info(f'Action space size: {game.getActionSize()} (M={game.M}, M^k={game.M}^{k})')
    log.info(f'Max steps per game: {game.max_steps}')

    nnet = NNet(game)

    if args.load_model:
        log.info(f'Loading model from {args.load_folder_file}')
        nnet.load_checkpoint(*args.load_folder_file)

    coach = Coach(game, nnet, args)

    if args.load_model:
        log.info("Loading training examples...")
        coach.loadTrainExamples()

    log.info('Starting AlphaZero training loop...')
    coach.learn()


if __name__ == '__main__':
    main()
