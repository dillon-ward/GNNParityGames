# gnn_pr_solver

## Create games and solutions

**Run:** 

python game_generator.py -n NUM_GAMES -gdir GAMES_DIR -sdir SOLUTIONS_DIR -pg PGSOLVER_BASE [-minn MIN_NODES]
                             [-maxn MAX_NODES] [-minrod MIN_ROD] [-maxrod MAX_ROD]

  -n: number of games
                        Categories to parse.
  -gdir: Directory to store games.
  -sdir: Directory to store solutions.
  -pg: Home directory of pg_solver. This is used to create and solve games.
  -minn: Minimum number of nodes
  -maxn: Maximum number of nodes.
  -minrod: Minimum relative outdegree (Minimum fraction of nodes each node points to).
  -maxrod: Maximum relative outdegree (Maximum fraction of nodes each node points to).

## Train models 

See Jupyter notebook for examples.
