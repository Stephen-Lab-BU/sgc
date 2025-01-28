#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=08:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

evalf='oracle'
init='oracle'

for seed in 0 1 2 3 4
do
    for L in 10 25 50 100 #150 200
    do
        for alpha in -2.0 -1.0 0.0 2.0 4.0
        do
            python simulate/simple_pp_log.py latent.L=$L latent.seed=$seed obs.alpha=$alpha obs.seed=$seed
            python fit/simple_model.py latent=single-freq-log obs=pp_log latent.L=$L latent.seed=$seed obs.alpha=$alpha obs.seed=$seed model.eigvals_flag=$evalf model.model_init=$init
        done
    done
done


