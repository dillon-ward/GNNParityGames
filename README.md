<!--
SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
SPDX-FileContributor: Tobias Hecking <tobias.hecking@dlr.de>
SPDX-FileContributor: Alexander Weinert <alexander.weinert@dlr.de>

SPDX-License-Identifier: CC-BY-NC-ND-3.0
-->

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/DLR-SC/GNN-Parity-Games-Solver/blob/main/LICENSES/MIT.txt)
[![REUSE status](https://api.reuse.software/badge/github.com/DLR-SC/GNN-Parity-Games-Solver)](https://api.reuse.software/info/github.com/DLR-SC/GNN-Parity-Games-Solver)
[![Badge: Citation File Format Inside](https://img.shields.io/badge/-citable%20software-green)](https://github.com/DLR-SC/GNN-Parity-Games-Solver/blob/main/CITATION.cff)

# gnn_pr_solver

## Requirements

    pip install --user torch torch-geometric

## Train models

Use

    ./gnn-pg-solver.py train --network GAT --output GAT_weights.pth games/game_1.txt solutions/solution_1.txt

or its equivalent using shorthand options

    ./gnn-pg-solver.py train -n GAT -o GAT_weights.pth games/game_1.txt solutions/solution_1.txt

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
    paste -d' ' results.csv solutions.csv | ./gnn-pg-solver.py evaluate 
