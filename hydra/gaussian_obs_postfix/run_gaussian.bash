#!/bin/bash -l

# next get working: empirical init, low-rank model
iters=50
# ov1=2.5
# for K in 3 10 25 
# do
#     gamma=k$K-chlg4-gaussian-rank1-nz9
#     for L in 10 25 50 
#     do 
#         for ov2 in -2 0 
#         do
#             init=empirical-init
#             python scripts/hydra_run_gaussian_rank1.py latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init obs.ov1=2.5 obs.ov2=$ov2 \
#             hydra.job.chdir=True
#         done
#     done
# done


mstep=low-rank
msteprank=1
ov1=2.5
K=3
gamma=k$K-chlg4-gaussian-rank1-nz9
L=10
ov2=-1
init=flat-init
scale_init=100
python scripts/hydra_run_gaussian_rank1.py latent.gamma=$gamma latent.L=$L \
    'model.support=[9,9]' model.emiters=$iters model.scale_init=$scale_init \
    model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 model.m_step_option=$mstep \
    model.m_step_rank=$msteprank hydra.job.chdir=True

# init=empirical-init
# python scripts/hydra_run_gaussian_rank1.py latent.gamma=$gamma latent.L=$L 'model.support=[9,9]' model.emiters=$iters model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 \
#     hydra.job.chdir=True

