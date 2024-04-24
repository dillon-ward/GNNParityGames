#!/bin/bash

model="$1"
shift

if [ "$model" == "" ]; then
    echo "no model specified"
    exit 1
fi

weights_file=${model}/weights.pth

mkdir "$model" 2>/dev/null

rm g s 2>/dev/null
ln -s games-small/games/ g
ln -s games-small/solutions/ s

rm -rf pg_data_20220708 "$weights_file" 2>/dev/null

for i in $(seq -w 0 2099)
do
    echo "g/game_$i.txt"
    echo "s/solution_game_$i.txt"
done | xargs python gnn-pg-solver.py train -n "$model" -o "$weights_file" || {
    rm -rf g s "$model" 2>/dev/null
    exit 1
}

rm g s 2>/dev/null
