#!/bin/bash -l

eigvalflag='fit'
# python plot/plot_lowrank1_fit.py 
init='empirical'
K=3
for drop_emp in true false
do
    for seed in 0 1 #2 3 4 
    do
        python plot/plot_lowrank1_fit_eigval.py latent=single_freq_log latent.K=$K obs=pp_log model.eigvals_flag=$eigvalflag \
        latent.seed=$seed obs.seed=$seed model.model_init=$init 'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]' \
        plot.drop_emp=$drop_emp
    done
done


# init='oracle'
# for seed in 0 1 2 3 4 
# do
#     python plot/plot_lowrank1_fit.py latent=single-freq-log obs=pp_log model.eigvals_flag=$eigvalflag \
#     latent.seed=$seed obs.seed=$seed model.model_init=$init 'plot.Ls=[10,25,50,100]' 'plot.thetas=[-2.0,-1.0,0.0,2.0,4.0]'
# done