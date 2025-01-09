#!/bin/bash -l

# for K in 3 #10
eigrank=1
for K in 3 
do
    for seed in 7 
        do
            for init in oracle-init empirical-init
            do
                python plots_rankRmodel.py $K $init $seed $eigrank
            done
        done
    done
done
