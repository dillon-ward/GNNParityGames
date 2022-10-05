<!--
SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
SPDX-FileContributor: Tobias Hecking <tobias.hecking@dlr.de>
SPDX-FileContributor: Alexander Weinert <alexander.weinert@dlr.de>

SPDX-License-Identifier: CC-BY-NC-ND-3.0
-->

<p align="center">
  <a href="https://github.com/dlr-sc/gitlab2prov/blob/master/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg" target="_blank" />
  </a>
  <a href="https://citation-file-format.github.io/">
    <img alt="Badge: REUSE compliant" src="https://img.shields.io/reuse/compliance/github.com/DLR-SC/gnn_pr_solver" target="_blank" />
  </a>
  <a href="https://citation-file-format.github.io/">
    <img alt="Badge: Citation File Format Inside" src="https://img.shields.io/badge/-citable%20software-green" target="_blank" />
  </a>
</p>

# gnn_pr_solver

## Requirements

    pip install --user torch torch_sparse torch-geometric torch-scatter wandb

## Create games and solutions

**Run:** 

    python game_generator.py -n NUM_GAMES -gdir GAMES_DIR -sdir SOLUTIONS_DIR -pg PGSOLVER_BASE [-minn MIN_NODES]
                             [-maxn MAX_NODES] [-minrod MIN_ROD] [-maxrod MAX_ROD]

- -n: number of games
- -gdir: Directory to store games.
- -sdir: Directory to store solutions.
- -pg: Home directory of pg_solver. This is used to create and solve games.
- -minn: Minimum number of nodes
- -maxn: Maximum number of nodes.
- -minrod: Minimum relative outdegree (Minimum fraction of nodes each node points to).
- -maxrod: Maximum relative outdegree (Maximum fraction of nodes each node points to).

## Train models 

Use

    ./gnn-pg-solver.py train --network GAT --output GAT_weights.pth games/game_1.txt solutions/soluation_1.txt

or its equivalent using shorthand options

    ./gnn-pg-solver.py train -n GAT -o GAT_weights.pth games/game_1.txt solutions/soluation_1.txt

## Predict winning regions

We assume that the directory `games` contains a set of plain-text files containing parity games.
Use

    ./gnn-pg-solver.py predict --network GAT --weights GAT_weights.pth --output results.csv games/*

or its equivalent using shorthand options

    ./gnn-pg-solver.py predict -n GAT -w GAT_weights.pth -o results.csv games/*

## Evaluate predictions

First, generate some predictions using the command in the previous section.
We assume that for each game `games/game_XXXX.txt` there exists a solution `solutions/solution_game_XXXX.txt`.

The `evaluate` subcommand expects the input to be in the form `game prediction reference`, so we need to add the final column to the results before handing them to the evaluation.

    sed 's:^games/game_\([0-9]*\).txt.*$:solutions/solution_game_\1.txt:' < results.csv > solutions.csv
    paste -d' ' results.csv solutions.csv | ./gnn-pg-solver evaluate 