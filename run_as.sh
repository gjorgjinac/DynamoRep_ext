#!/bin/bash

for train_alg in "DE" "ES" "PSO"
do
    for dimension in 5 
    do
        for split_type in "P" "I"
        do
            for iteration in 0 2 4 9 19 29
            do
     
                tsp python algorithm_selection_iteration_ela.py DE-ES-PSO $split_type true 0-$iteration $dimension true $train_alg
                tsp python algorithm_selection_iteration_ela.py DE-ES-PSO $split_type true 0-$iteration $dimension false $train_alg
                tsp python algorithm_selection_iteration_ela.py DE-ES-PSO $split_type false 0-$iteration $dimension true $train_alg
                tsp python algorithm_selection_iteration_ela.py DE-ES-PSO $split_type false 0-$iteration $dimension false $train_alg
                tsp python algorithm_selection.py DE-ES-PSO $split_type true 0-$iteration $dimension true $train_alg true
                tsp python algorithm_selection.py DE-ES-PSO $split_type false 0-$iteration $dimension true $train_alg true
                tsp python algorithm_selection.py DE-ES-PSO $split_type true 0-$iteration $dimension false $train_alg true
                tsp python algorithm_selection.py DE-ES-PSO $split_type false 0-$iteration $dimension false $train_alg true
            done
        done
    done
done
