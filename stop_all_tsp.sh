#!/bin/bash

for i in {0..100}
do
    #tsp -u $i
    tsp -r $i
    kill $(tsp -p $i)
done