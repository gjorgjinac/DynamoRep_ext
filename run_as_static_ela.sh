#!/bin/bash


for dimension in 5 
do
    for split_type in "P" "I"
    do
        for scdf in 10 30 50 100 200 300
        do

            tsp python algorithm_selection_ela_static.py DE-ES-PSO $split_type $dimension true $scdf
            tsp python algorithm_selection_ela_static.py DE-ES-PSO $split_type $dimension false $scdf
        done
    done
done



