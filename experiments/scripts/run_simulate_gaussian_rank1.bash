#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=04:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env


gseed=0
K=3
ov1=1
for seed in {0..19}
do
    for L in 3 5 10 25
    do
        # for mu in -2.0 -1.0 0.0 2.0 4.0
        # for mu in -2.0 -1.0 0.0 2.0 
        for ov2 in -3.0 -2.0 -1.0 0.0
        do
            python ../simulate/simulate_simple.py latent=single_freq obs=gaussian latent.gamma_seed=$gseed latent.K=$K latent.L=$L latent.seed=$seed obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed
        done
    done
done

# seed=42
# L=27
# mu=1.9
# python simulate/simulate_simple.py latent=single_freq_log obs=pp_log latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed