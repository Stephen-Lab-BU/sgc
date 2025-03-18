#!/bin/bash -l

# eigvalflag='fit'
# # python plot/plot_lowrank1_fit.py 
# init='empirical'
# K=3
# for seed in 0 1 #2 3 4 
# do
#     python plot/plot_lowrank1_fit.py latent=single_freq_log latent.K=$K obs=pp_log model.eigvals_flag=$eigvalflag \
#     latent.seed=$seed obs.seed=$seed model.model_init=$init 'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]'
# done

K=3
init='empirical'
scale_init=1000000
for seed in 1 2 3 4 
do
    python plot/plot_fullrank_fit_eigval.py latent=single_freq_log obs=pp_log model=fullrank \
    latent.seed=$seed obs.seed=$seed \
    model.model_init=$init 'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]'
    # model.model_init=$init model.scale_init=$scale_init 'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]'
done