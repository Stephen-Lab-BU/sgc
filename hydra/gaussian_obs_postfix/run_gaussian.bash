#!/bin/bash -l

iters=50
for K in 3 
do
    gamma=k$K-chlg3-gaussian-rank1-nz9
    for L in 10 25
    do 
        for ov2 in -2 0 2
        do
            init=flat-init
            scale_init=100000
            python scripts/hydra_run_gaussian_rank1.py latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init model.scale_init=$scale_init obs.ov2=$ov2 \
            hydra.job.chdir=True

            # init=empirical-init
            # python scripts/hydra_run_gaussian_rank1.py latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init  obs.ov2=$ov2 \
            # hydra.job.chdir=True
        done
    done
done

# # iters=50
# # for gamma in 'k3-chlg3-rank1-nz9' 'k10-chlg3-rank1-nz9' 'k25-chlg3-rank1-nz9' 
# for K in 3 10 25
# do
#     gamma=k$K-chl3-rank1-rank1-nz9
#     echo $gamma
# done
