# emiters=20

# for L in 50 #100 250 
# do 
#     # for alpha in 0 50 100 300
#     for alpha in 200 
#     do 
#         python hydra_run_pp_relu.py latent.L=$L 'model.support=[0,50]' model.emiters=$emiters obs.alpha=$alpha \
#         hydra.job.chdir=True
#     done
# done

#!/bin/bash -l

# Request a parallel environment with 8 cores 
#$ -pe omp 4
#$ -l h_rt=03:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env


for m_step_mod in double-Ups-conj-hess 
do
    for init in flat-init
    do
        for L in 50
        do 
            for alpha in 50 200 300
            do
                # for scale_init in 10 1000 100000
                for scale_init in 100000 10000000 1000000000
                do
                    python ../hydra_run_pp_relu.py latent.L=$L 'model.support=[0,50]' model.m_step_mod=$m_step_mod model.emiters=20 model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
                    hydra.job.chdir=True
                    # python hydra_run_pp_relu.py latent.L=$L 'model.support=[0,50]' model.emiters=20 model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
                    # hydra.job.chdir=True
                    # for method in old oldmod # jax
                    # do
                    #     python hydra_run_pp_relu_ts.py latent.L=$L model.ts_method=$method 'model.support=[0,50]' model.emiters=20 model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
                    #     hydra.job.chdir=True
                    # done
                    # for ic in true false
                    # do
                    #     python hydra_run_pp_relu_scipy.py latent.L=$L model.inverse_correction=$ic 'model.support=[0,50]' model.emiters=20 model.init=$init model.scale_init=$scale_init obs.alpha=$alpha \
                    #     hydra.job.chdir=True
                    # done
                done
            done
        done
    done
done