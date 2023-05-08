#!/bin/bash

for end_iteration in 0 2 4 9 19 29
do

    for algorithm in "PSO" "ES" "DE"
    do
        for dimension in 3 5 10 20
        do
            #tsp -L SINGLESTAT python problem_classification.py $algorithm true true 0-$end_iteration $dimension true
            #tsp -L SINGLESTAT python problem_classification.py $algorithm true true 0-$end_iteration $dimension false
            #tsp -L SINGLESTAT python problem_classification_single_statistic.py $algorithm true false 0-$end_iteration $dimension true
            #tsp -L SINGLESTAT python problem_classification_single_statistic.py $algorithm true false 0-$end_iteration $dimension false
            
            tsp -L SINGLESTAT python problem_classification_x_y.py $algorithm true false 0-$end_iteration $dimension true
            tsp -L SINGLESTAT python problem_classification_x_y.py $algorithm true false 0-$end_iteration $dimension false
        done
    done
done