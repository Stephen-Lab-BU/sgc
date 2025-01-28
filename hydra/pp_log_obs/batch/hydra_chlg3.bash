#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=04:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

iters=50
K=10
gamma=k$K-log-chlg3-rank1-nz9
for s in 8 9 10
do
    for L in 10 25 
    do 
        # for alpha in 1.5 2.0 2.5
        for alpha in 1.0 2.0
        do
            init=flat-init
            scale_init=100000
            python ../scripts/hydra_run_pp_log_rank1.py latent.seed=$s obs.seed=$s latent.gamma=$gamma latent.L=$L 'model.support=[9,9]' model.emiters=$iters model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
            hydra.job.chdir=True
        done
    done
done

L=10
for s in 8 9 10 
do 
    # for alpha in 1.5 2.0 2.5
    for alpha in 1.0 2.0
    do
        init=flat-init
        scale_init=100000
        python ../scripts/hydra_run_pp_log_rank1.py obs.seed=$s latent.gamma=$gamma latent.L=$L 'model.support=[9,9]' model.emiters=$iters model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
        hydra.job.chdir=True
    done
done