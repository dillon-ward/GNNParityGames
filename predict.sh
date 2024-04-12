#!/bin/bash

model="$1"
shift

if [ "$model" == "" ]; then
    echo "no model specified"
    exit 1
fi
 
weights_file=${model}/weights.pth

FIRST=0
LAST=2099

set="$1"
shift

if [ "$set" = "test" ] 
then
    FIRST=2100
    LAST=2999
fi

results_file=${model}/${set}_results.txt

rm g 2>/dev/null
ln -s games-small/games/ g

for i in $(seq -w "$FIRST" "$LAST")
do
    echo "g/game_$i.txt"
done | xargs python gnn-pg-solver.py predict -n $model -w "$weights_file" -o "$results_file"

rm g 2>/dev/null
