#!/bin/bash -l

for K in 3 10
do
    # for seed in 7 8 9 
    for seed in 9  
        do
            init=all
            python plots_rank1model_full.py $K $init $seed
        done
    done
done
