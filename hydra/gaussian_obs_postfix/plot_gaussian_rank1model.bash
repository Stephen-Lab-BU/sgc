#!/bin/bash -l

# for K in 3 #10
for K in 3 
do
    for seed in 7 
        do
            for init in oracle-init #empirical-init
            do
                python plots_rank1model.py $K $init $seed
            done
        done
    done
done
