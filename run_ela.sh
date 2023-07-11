#!/bin/bash

for problem_id in {1..24}
do
    for dimension in 5
    do
        for seed in 200 400 600 800 1000
        do
            for algorithm in "PSO" "ES" "DE"
            do
                tsp -L ELA Rscript ela_iteration_2.R $algorithm $seed $dimension $problem_id
                tsp -L ELA Rscript ela_iteration_2_normalized.R $algorithm $seed $dimension $problem_id
            done
        done
    done
done
