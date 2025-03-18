#!/bin/bash -l

for K in 3 10 
do
    for seed in 7
        do
            init=all
            python plots_rank1naive.py $K $init $seed
        done
    done
done
