#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=08:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

evalf='oracle'
init='flat'

# for seed in 0 1 2 3 4
# do
#     for L in 5 10 25 50 100 150 200
#     do
#         for mu in -2.0 -1.0
#         do
#             python simulate/simulate_simple.py latent=single_freq_log obs=pp_log latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed
#             python fit/fit_simple.py latent=single_freq_log obs=pp_log latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed model.eigvals_flag=$evalf model.model_init=$init
#         done
#     done
# done

# for seed in 0 1 2 3 4
for seed in 2
do
    # for L in 5 10 25 50 100 150 200
    for L in 25 
    do
        # for mu in -2.0 -1.0
        for mu in 2.0
        do
            python simulate/simulate_simple.py latent=single_freq_log obs=pp_log latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed
            python fit/fit_simple.py latent=single_freq_log obs=pp_log latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed model.eigvals_flag=$evalf model.model_init=$init
        done
    done
done

# seed=42
# L=27
# mu=1.9
# python simulate/simulate_simple.py latent=single_freq_log obs=pp_log latent.L=$L latent.seed=$seed obs.mu=$mu obs.seed=$seed