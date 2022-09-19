<!---
SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)

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

See Jupyter notebook for examples.
