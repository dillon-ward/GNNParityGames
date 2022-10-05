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

    for i in {games/game_1.txt,solutions/solution_game_1.txt}; do cat $i; echo ---; done | ./gnn-pg-solver.py train --network GAT > weights.ipt

