#!/bin/bash -l

iters=25
seed=7
ts_flag=fixed_eigval_true
ts_flag2=gd
m_step_init='warm_start'
for K in 3
do
    gamma=k$K-chlg4-rotate-gaussian-rank1-nz9
    # for L in 10 25 50
    for L in 10 25 
    do 
        for ov1 in 1.0 5.0
        do
            for ov2 in 0 
            # for ov2 in -1 0 
            do
                # for init in empirical-init 
                # for init in empirical-init true-init oracle-init
                for init in oracle-init empirical-init
                do 
                    python scripts/hydra_run_gaussian_rankR_model.py latent.seed=$seed obs.seed=$seed \
                    latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' \
                    model.ts_flag2=$ts_flag2 model.ts_flag=$ts_flag model.emiters=$iters \
                    model.m_step_init=$m_step_init model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 \
                    hydra.job.chdir=True hydra.run.dir='outputs/${now:%Y-%m-%d}-rankR/${model.ts_flag}-${model.ts_flag2}-rotatem/${latent.gamma}/seed-${latent.seed}/${model.m_step_init}/${now:%H-%M-%S}'
                done
            done
        done
    done
done



# iters=5
# ov1=5.0
# K=3
# gamma=k$K-chlg4-gaussian-rank1-nz9
# L=5
# ov2=-1
# init='flat-init'
# # python scripts/hydra_run_gaussian_rank1_model.py latent.gamma=$gamma latent.L=$L \
# #     'model.support=[9,9]' model.emiters=$iters model.scale_init=$scale_init \
# #     model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 \
# #     hydra.job.chdir=True
# scale_init=100
# python scripts/hydra_run_gaussian_rank1_model.py latent.gamma=$gamma latent.L=$L \
#     'model.support=[9,9]' model.emiters=$iters model.scale_init=$scale_init \
#     model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 \
#     hydra.job.chdir=True

# mstep=low-rank
# msteprank=1
# ov1=2.5
# K=3
# gamma=k$K-chlg4-gaussian-rank1-nz9
# L=10
# ov2=-1
# init=flat-init
# scale_init=100
# python scripts/hydra_run_gaussian_rank1.py latent.gamma=$gamma latent.L=$L \
#     'model.support=[9,9]' model.emiters=$iters model.scale_init=$scale_init \
#     model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 model.m_step_option=$mstep \
#     model.m_step_rank=$msteprank hydra.job.chdir=True

# init=empirical-init
# python scripts/hydra_run_gaussian_rank1.py latent.gamma=$gamma latent.L=$L 'model.support=[9,9]' model.emiters=$iters model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 \
#     hydra.job.chdir=True

