#!/bin/bash

model="$1"
shift

if [ "$model" == "" ]; then
    echo "no model specified"
    exit 1
fi

solutions_file=${model}/solutions.txt

set="$1"
shift

if [ "$set" == "" ]; then
    echo "no set specified"
    exit 1
fi

results_file=${model}/${set}_results.txt

eval_file=${model}/${set}_eval.txt

sed 's:^.*game_\([0-9]*\).txt.*$:games-small\/solutions\/solution_game_\1.txt:' < "$results_file" > "$solutions_file"
paste -d' ' "$results_file" "$solutions_file" | python gnn-pg-solver.py evaluate -o "$eval_file" --histogram

rm "$solutions_file" 2>/dev/null