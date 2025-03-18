#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=04:15:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

evalf='fit'
init='empirical'

for seed in {0..19}
do
    for L in 3 5 10 25
    do
        # for mu in -2.0 -1.0 0.0 2.0 4.0
        # for mu in -2.0 0.0 2.0 
        for mu in -1.0
        do
            # python ../simulate/simulate_simple.py latent=single_freq_log obs=pp_log latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed
            python ../fit/fit_simple_lowrank.py latent=single_freq_log obs=pp_log model=lowrank_eigh latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed model.eigvals_flag=$evalf model.model_init=$init 
        done
    done
done