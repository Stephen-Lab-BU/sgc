#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=04:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

iters=25

for k in 3 10
do
    gamma=k$K-chlg3-relu-rank1-nz9
    for L in 10 25 50
    do 
        for mu in 10 25 50
        do
            init=flat-init
            scale_init=10000000
            python ../scripts/hydra_run_pp_relu_rank1_model.py latent.seed=$seed obs.seed=$seed latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init model.scale_init=$scale_init obs.alpha=$mu \
            hydra.job.chdir=True 

            init=empirical-init
            python ../scripts/hydra_run_pp_relu_rank1_model.py latent.seed=$seed obs.seed=$seed latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init obs.alpha=$alpha \
            hydra.job.chdir=True 
        done
    done
done