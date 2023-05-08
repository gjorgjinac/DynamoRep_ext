#!/bin/bash
tsp -S 10
for end_iteration in 0 2 4 9 19 29 39 49
do

    for algorithm in "PSO" "DE" "CMAES" "ES"
    do
        for dimension in 3 5 10 20
        do
            for predict_iteration in 2 4 9 19 29 39 49
            do
                if [ $predict_iteration -gt $end_iteration ]
                then
                    tsp python performance_prediction_different_iterations.py $algorithm false 0-$end_iteration $predict_iteration false $dimension
                fi
            done
        done
    done
done