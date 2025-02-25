#!/bin/bash -l

eigvalflag='oracle'
# python plot/plot_lowrank1_fit.py 
init='flat'
K=9
for seed in 0 1 2 3 4 
do
    python plot/plot_lowrank1_fit.py latent=single_freq_log latent.K=$K obs=pp_log \
    model.eigvals_flag=$eigvalflag latent.seed=$seed obs.seed=$seed model.model_init=$init \
    'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]' \
    'plot.dims=[0,1,2,3,4,5,6,7,8]'
done

# init='oracle'
# for seed in 0 1 2 3 4 
# do
#     python plot/plot_lowrank1_fit.py latent=single-freq-log obs=pp_log model.eigvals_flag=$eigvalflag \
#     latent.seed=$seed obs.seed=$seed model.model_init=$init 'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]'
# done