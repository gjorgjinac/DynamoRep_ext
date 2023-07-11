#!/bin/bash

for end_iteration in 0 2 4 9 19 29
do

    for dimension in 3 5 10 20
    do
        tsp -L ALGORITHM python algorithm_classification.py DE-PSO-ES true false 0-$end_iteration $dimension false
        tsp -L ALGORITHM python algorithm_classification.py DE-PSO-ES true false 0-$end_iteration $dimension true
        
        if [[ $end_iteration -gt 0 ]]
        then
            tsp -L ALGORITHM python algorithm_classification.py DE-PSO-ES true true 0-$end_iteration $dimension true
            tsp -L ALGORITHM python algorithm_classification.py DE-PSO-ES true true 0-$end_iteration $dimension false
        fi
    done

done