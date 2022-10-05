#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
# SPDX-FileContributor: Tobias Hecking <tobias.hecking@dlr.de>
# SPDX-FileContributor: Alexander Weinert <alexander.weinert@dlr.de>
#
# SPDX-License-Identifier: MIT

import argparse
from torch_geometric.loader import DataLoader
from parity_game_dataset import ParityGameDataset
from parity_game_network import ParityGameGCNNetwork, ParityGameGATNetwork
import math
import torch
import numpy as np
import sys

import parity_game_network as pn

class Training:

    def train(self, games, solutions):
        data = ParityGameDataset('pg_data_20220708', games, solutions)
        train_loader = DataLoader(data, batch_size=5, shuffle=True)

        self._train_internal(train_loader)
        train_acc_nodes, train_acc_edges = self._test(train_loader)

        torch.save(self._network.state_dict(), self._output_stream)

    def _train_internal(self, loader):
        running_loss = 0.
        i = 0

        optimizer = torch.optim.Adam(self._network.parameters(), lr=0.001)
        criterion_nodes = torch.nn.CrossEntropyLoss()
        criterion_edges = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))

        self._network.train()

        for data in loader:  # Iterate in batches over the training dataset.
            i += 1
            optimizer.zero_grad()  # Clear gradients.
            out_nodes, out_edges = self._network(data.x, data.edge_index)  # Perform a single forward pass.

            # Most edges do not belong to a winning strategy and thus the data is extemely imbalanced. The model will probably learn that predicting always "non-winning" for each edge
            # yields reasonable performance. To avoid this, approximately as many non-winning strategy edges are sampled as there are winning edges.
            # edge_selection = (torch.rand(data.y_edges.shape[0]) > 0.7) | (data.y_edges == 1)
            loss = criterion_nodes(out_nodes, data.y_nodes) + criterion_edges(out_edges,
                                                                              data.y_edges)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

            running_loss += loss.item()
            if i % 50 == 0:
                last_loss = running_loss / 50  # loss per batch
                running_loss = 0.

    def _test(self, loader):
        self._network.eval()

        correct_vertices = 0

        for data in loader:  # Iterate in batches over the training/test dataset.
            out_nodes, _ = self._network(data.x, data.edge_index)
            pred_nodes = out_nodes.argmax(dim=1)  # Use the class with highest probability.
            correct_vertices += (pred_nodes == data.y_nodes).sum() / len(
                pred_nodes)  # Check against ground-truth labels.
        return correct_vertices / len(loader)  # Derive ratio of correct predictions.

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def output(self):
        return self._output_stream

    @output.setter
    def output(self, value):
        self._output_stream = value

class Solver:
    pass

def train(args):
    training = Training()

    if args.network == "GAT":
        training.network = ParityGameGATNetwork(256, 256, 10)
    elif args.network == "GCN":
        training.network = ParityGameGCNNetwork(256, 256, 10)

    if args.output is None or args.output == '-':
        training.output = sys.stdout.buffer
    else:
        training.output = args.output

    if len(args.files) == 0 or (len(args.files) == 1 and args.files[0] == '-'):
        games, solutions = [], []
        current_list, next_list = games, solutions
        buffer = []
        for line in sys.stdin.readlines():
            print(line)
            if line.startswith('---'):
                current_list.append(buffer)
                current_list, next_list = next_list, current_list
                buffer = []
            else:
                buffer.append(line)
    elif len(args.files) % 2 == 0:
        games, solutions = [], []
        for i in range(len(args.files)):
            with open(args.files[i], 'r') as f:
                if i % 2 == 0:
                    games.append(f.readlines())
                else:
                    solutions.append(f.readlines())
    else:
        raise ValueError("Invalid list of files given. List of files must either be empty, contain only '-', or contain an even number of file names, alternating between games and solutions, starting with games")

    print(games)
    print(solutions)

    training.train(games, solutions)

    print("Training")

def solve(args):
    Solver()
    print("Solving")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', required=True)
    train_parser = subparsers.add_parser('train', help='create a weights-file by training the solver on a set of solved games')
    train_parser.add_argument('-n', '--network', type=str, choices=['GCN','GAT'], required=True, help="The GNN architecture to train")
    train_parser.add_argument('-o', '--output', type=str, help="Where to write the output file. If - or omitted, output is written to stdout")
    train_parser.add_argument('files', nargs='*', help="A list of files containing games and solutions for training. Must alternate between games and solutions, starting with games. If empty or '-', files are read from stdin and expected to be separated by newlines")
    train_parser.set_defaults(func=train)

    solve_parser = subparsers.add_parser('solve', help='solve one or more parity games using a previously created weights-file')
    solve_parser.set_defaults(func=solve)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()