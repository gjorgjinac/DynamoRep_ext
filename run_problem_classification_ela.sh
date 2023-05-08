#!/bin/bash

for end_iteration in 0 2 4 9 19 29
do

    for algorithm in "DE"
    do
        for dimension in 5
        do
            tsp -L ELA python problem_classification_ela.py $algorithm true false $end_iteration $dimension
            tsp -L ELA python problem_classification_ela.py $algorithm true true $end_iteration $dimension
        done
    done
done