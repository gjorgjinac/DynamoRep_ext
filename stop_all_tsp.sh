#!/bin/bash

for i in {6000..6800}
do
    #tsp -u $i
    tsp -r $i
    kill $(tsp -p $i)
done