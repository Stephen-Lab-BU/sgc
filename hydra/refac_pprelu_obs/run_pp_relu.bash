#!/bin/bash -l

iters=3

seed=7
for K in 3 10
do
    gamma=k$K-chlg3-relu-rank1-nz9
    for L in 10 25 50
    do 
        for mu in 10 25 50
        do
            init=flat-init
            scale_init=10000000
            python scripts/hydra_run_pp_relu_rank1.py latent.seed=$seed obs.seed=$seed latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init model.scale_init=$scale_init obs.alpha=$mu \
            hydra.job.chdir=True hydra.run.dir='outputs/${now:%Y-%m-%d}/naive/seed-${latent.seed}/${now:%H-%M-%S}'

            init=empirical-init
            python scripts/hydra_run_pp_relu_rank1.py latent.seed=$seed obs.seed=$seed latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init obs.alpha=$mu \
            hydra.job.chdir=True hydra.run.dir='outputs/${now:%Y-%m-%d}/naive/seed-${latent.seed}/${now:%H-%M-%S}'
        done
    done
done


# iters=3
# K=3
# gamma=k$K-chlg3-relu-rank1-nz9
# for L in 10 25 50
# do 
#     for alpha in 10 25 50
#     do
#         init=flat-init
#         scale_init=10000000
#         python scripts/hydra_run_pp_relu_rank1_model.py latent.seed=8 latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
#         hydra.job.chdir=True hydra.run.dir='outputs/seed-${latent.seed}/${now:%Y-%m-%d}/${now:%H-%M-%S}'

#         # init=empirical-init
#         # python scripts/hydra_run_pp_relu_rank1_model.py latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init obs.alpha=$alpha \
#         # hydra.job.chdir=True 
#     done
# done

# # gamma=k$K-chlg3-rank1-nz9
# # for s in 8 9 10 11 12
# # do
# #     for L in 10 
# #     do 
# #         for alpha in 10 25 50
# #         do
# #             init=flat-init
# #             scale_init=10000000
# #             python scripts/hydra_run_pp_relu_rank1.py latent.seed=$s obs.seed=$s latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
# #             hydra.job.chdir=True

# #             # init=empirical-init
# #             # python scripts/hydra_run_pp_relu_rank1.py latent.gamma=$gamma latent.L=$L model.maxiter=10 'model.support=[9,9]' model.emiters=$iters model.init=$init obs.alpha=$alpha \
# #             # hydra.job.chdir=True
# #         done
# #     done
# # done