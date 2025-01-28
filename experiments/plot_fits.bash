#!/bin/bash -l

# eigvalflag='oracle'

# # python plot/plot_lowrank1_fit.py 
# for seed in 0 1  
# do
#     python plot/plot_lowrank1_fit.py latent=single-freq-relu obs=pp_relu model.eigvals_flag=$eigvalflag \
#     latent.seed=$seed obs.seed=$seed 'plot.Ls=[10,25,50,100]' 'plot.thetas=[10,30,50,100]'
# done


eigvalflag='oracle'
# python plot/plot_lowrank1_fit.py 
init='flat'
for seed in 0 1 2 3 4 
do
    python plot/plot_lowrank1_fit.py latent=single-freq-log obs=pp_log model.eigvals_flag=$eigvalflag \
    latent.seed=$seed obs.seed=$seed model.model_init=$init 'plot.Ls=[10,25,50,100,150,200]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]'
done

init='oracle'
for seed in 0 1 2 3 4 
do
    python plot/plot_lowrank1_fit.py latent=single-freq-log obs=pp_log model.eigvals_flag=$eigvalflag \
    latent.seed=$seed obs.seed=$seed model.model_init=$init 'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]'
done