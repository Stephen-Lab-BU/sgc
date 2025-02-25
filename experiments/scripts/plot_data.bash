#!/bin/bash -l

# python plot/plot_lowrank1_fit.py 

# for ov2 in -1.0 0.0
# do
#     python plot/plot_lowrank1_data.py plot.ov2=$ov2
# done

# L=50
# for seed in 0 1 
# do
#     for alpha in 10 30 50 100
#     do
#         python plot/plot_lowrank1_data_pprelu.py plot.alpha=$alpha latent.L=$L latent.seed=$seed obs.seed=$seed
#     done
# done


L=5
for seed in 0 1 2 3 4
do
    # for alpha in 0.0 1.0 2.0 3.0 4.0 5.0
    for alpha in -2.0 -1.0
    do
        python plot/plot_lowrank1_data_pplog.py plot.alpha=$alpha plot.L=$L latent.seed=$seed obs.seed=$seed
    done
done