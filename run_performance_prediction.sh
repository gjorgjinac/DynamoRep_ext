#!/bin/bash
tsp -S 10
for end_iteration in 0 2 4 9 19 29 39 49
do

    for algorithm in "PSO" "DE" "CMAES" "ES"
    do
        for dimension in 3 5 10 20
        do
            tsp python performance_prediction.py $algorithm false 0-$end_iteration false $dimension
            tsp python performance_prediction.py $algorithm false 0-$end_iteration true $dimension
            tsp python performance_prediction.py $algorithm true 0-$end_iteration false $dimension
            tsp python performance_prediction.py $algorithm true 0-$end_iteration true $dimension
        done
    done
done