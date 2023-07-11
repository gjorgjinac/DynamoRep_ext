#!/bin/bash
tsp -S 10
for seed in 200 400 600 800 1000
do
    for dimension in 3 5 10 20
    do
        tsp python algorithms.py $seed $dimension
    done
done
