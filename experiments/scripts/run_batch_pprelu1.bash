#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=08:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

evalf='oracle'

for L in 10 25 50 100
do
    for alpha in 100
    do
        for seed in 0 1  
        do
            python simulate/simulate_simple.py latent=single_freq_relu obs=pp_relu latent.L=$L latent.seed=$seed obs.alpha=$alpha obs.seed=$seed
            # python fit/simple_model.py latent=single-freq-relu obs=pp-relu latent.L=$L latent.seed=$seed obs.alpha=$alpha obs.seed=$seed model.eigvals_flag=$evalf
        done
    done
done


# for seed in 0 1  
# do
#     for L in 10 25
#     do
#         for alpha in 10 30 50 
#         do
#             python simulate/simple_pp_relu.py latent.L=$L latent.seed=$seed obs.alpha=$alpha obs.seed=$seed
#             python fit/simple_model.py latent=single-freq-relu obs=pp_relu latent.L=$L latent.seed=$seed obs.alpha=$alpha obs.seed=$seed model.eigvals_flag=$evalf
#         done
#     done
# done


