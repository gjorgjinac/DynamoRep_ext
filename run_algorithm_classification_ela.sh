#!/bin/bash

for end_iteration in 0 2 4 9 19 29
do


    for dimension in 5
    do
        #tsp -L ALG python algorithm_classification_ela.py DE-PSO-ES true false 0-$end_iteration $dimension true
        tsp -L ALG python algorithm_classification_ela.py DE-PSO-ES true true 0-$end_iteration $dimension true
        tsp -L ALG python algorithm_classification_ela.py DE-PSO-ES true true 0-$end_iteration $dimension false
        tsp -L ALG python algorithm_classification_ela.py DE-PSO-ES true false 0-$end_iteration $dimension true
        tsp -L ALG python algorithm_classification_ela.py DE-PSO-ES true false 0-$end_iteration $dimension false
    done

done