#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=04:15:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

gseed=0
evalf='fit'
init='empirical'
ov1=1

for seed in {0..2}
do
    # for L in 5 25
    for L in 5 25 
    do
        for ov2 in -2.0 -1.0 0.0 1.0 2.0
        do
            python ../simulate/simulate_simple.py latent=single_freq obs=gaussian latent.gamma_seed=$gseed latent.L=$L latent.seed=$seed \
                obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed
            python ../fit/fit_model.py latent=single_freq obs=gaussian model=lowrank_eigh latent.L=$L latent.seed=$seed \
                obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed model.eigvals_flag=$evalf model.model_init=$init 
            python ../fit/fit_model.py latent=single_freq obs=gaussian model=fullrank latent.L=$L latent.seed=$seed \
                obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed model.model_init=$init 
            # python ../fit/fit_model.py latent=single_freq obs=gaussian model=fullrank_pinv latent.L=$L latent.seed=$seed \
            #     obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed model.model_init=$init 
        done
    done
done