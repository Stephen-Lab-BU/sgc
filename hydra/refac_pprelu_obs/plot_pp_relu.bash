#!/bin/bash -l

for K in 3 10 
do
    for seed in 7
        do
            init=all
            python plots_rank1model.py $K $init $seed
        done
    done
done
