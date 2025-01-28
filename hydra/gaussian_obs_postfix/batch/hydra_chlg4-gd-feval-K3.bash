#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=02:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

iters=25
seed=7
ts_flag=fixed_eigval_true
ts_flag2=gd
m_step_init=warm_start
for K in 3
do
    gamma=k$K-chlg4-rotate-gaussian-rank1-nz9
    for L in 10 25 50
    do 
        for ov1 in 1.0 5.0
        do
            for ov2 in -1 0 
            do
                # for init in empirical-init 
                # for init in empirical-init true-init oracle-init
                for init in oracle-init empirical-init
                do 
                    python ../scripts/hydra_run_gaussian_rankR_model.py latent.seed=$seed obs.seed=$seed \
                    latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' \
                    model.ts_flag2=$ts_flag2 model.ts_flag=$ts_flag model.emiters=$iters \
                    model.m_step_init=$m_step_init model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 \
                    hydra.job.chdir=True hydra.run.dir='outputs/${now:%Y-%m-%d}-rankR/${model.ts_flag}-${model.ts_flag2}-rotatem/${latent.gamma}/seed-${latent.seed}/${model.m_step_init}/${now:%H-%M-%S}'
                done
            done
        done
    done
done

# iters=25
# seed=7
# ts_flag=fixed_eigval_true

# for seed in 7 8 9
# do 
#     for K in 3
#     do
#         gamma=k$K-chlg4-rotate-gaussian-rank1-nz9
#         for L in 10 25 50
#         # for L in 10 25 
#         do 
#             for ov1 in 1.0 5.0
#             do
#                 # for ov2 in 0 
#                 for ov2 in -1 0 
#                 do
#                     init=flat-init
#                     scale_init=100
#                     python ../scripts/hydra_run_gaussian_rank1_model.py latent.seed=$seed obs.seed=$seed \
#                     latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' \
#                     model.ts_flag=$ts_flag model.emiters=$iters model.init=$init model.scale_init=$scale_init \
#                     obs.ov1=$ov1 obs.ov2=$ov2 \
#                     hydra.job.chdir=True hydra.run.dir='outputs/${now:%Y-%m-%d}-${model.ts_flag}/${latent.gamma}/seed-${latent.seed}/${now:%H-%M-%S}'
#                     init=empirical-init
#                     python ../scripts/hydra_run_gaussian_rank1_model.py latent.seed=$seed obs.seed=$seed \
#                     latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' \
#                     model.ts_flag=$ts_flag model.emiters=$iters model.init=$init obs.ov1=$ov1 obs.ov2=$ov2 \
#                     hydra.job.chdir=True hydra.run.dir='outputs/${now:%Y-%m-%d}-${model.ts_flag}/${latent.gamma}/seed-${latent.seed}/${now:%H-%M-%S}'
#                 done
#             done
#         done
#     done
# done