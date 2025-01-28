#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=08:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

for L in 10 25 50
do
    for ov1 in 1.0 5.0
    do
        for ov2 in -2.0 -1.0 
        do
            for seed in 0 1 2 3 4 
            do
                python simulate/simple_gaussian.py latent.L=$L latent.seed=$seed obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed
                python fit/simple_model.py latent.L=$L latent.seed=$seed obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed
            done
        done
    done
done

