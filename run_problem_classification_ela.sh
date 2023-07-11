#!/bin/bash

for end_iteration in 0 2 4 9 19 29
do

    for algorithm in "DE" "ES" "PSO"
    do
        for dimension in 5
        do
            tsp -L ELA python problem_classification_ela.py $algorithm true false 0-$end_iteration $dimension true
            tsp -L ELA python problem_classification_ela.py $algorithm true true 0-$end_iteration $dimension true
            tsp -L ELA python problem_classification_ela.py $algorithm true false 0-$end_iteration $dimension false
            tsp -L ELA python problem_classification_ela.py $algorithm true true 0-$end_iteration $dimension false
        done
    done
done