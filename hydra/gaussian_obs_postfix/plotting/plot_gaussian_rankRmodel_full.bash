#!/bin/bash -l

# for K in 3 #10
eigrank=1
for K in 3 10
do
    for seed in 8 9 10
        do
            for init in oracle-init empirical-init
            do
                python plots_rankRmodel_full.py $K $init $seed $eigrank
            done
        done
    done
done
