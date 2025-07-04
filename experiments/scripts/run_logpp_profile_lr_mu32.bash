#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=04:00:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

gseed=0
evalf='fit'
init='empirical'
# L=25
scalepower=50000
emiters=20
mu=3.2
K=10
L=25

for seed in {100..200}
do
    python ../simulate/simulate_simple.py latent=single_freq_log obs=pp_log \
        latent.gamma_seed=$gseed latent.K=$K latent.L=$L latent.scale_power_target=$scalepower \
        latent.seed=$seed obs.mu=$mu obs.seed=$seed
    # python ../fit/fit_model.py latent=single_freq_log obs=pp_log model=fullrank latent.K=$K latent.L=$L latent.seed=$seed \
    #     obs.mu=$mu obs.seed=$seed model.model_init=$init 
    # PINV
    python ../fit/fit_model.py latent=single_freq_log obs=pp_log model=fullrank_pinv \
        latent.K=$K latent.L=$L latent.seed=$seed latent.scale_power_target=$scalepower \
        obs.mu=$mu obs.seed=$seed model.model_init=$init model.num_em_iters=$emiters \
        model.m_step_option=testing-factor-igprior
done

        # do
        #     python ../simulate/simulate_simple.py latent=single_freq obs=gaussian latent.gamma_seed=$gseed latent.L=$L latent.seed=$seed \
        #         obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed
        #     python ../fit/fit_model.py latent=single_freq obs=gaussian model=lowrank_eigh latent.L=$L latent.seed=$seed \
        #         obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed model.eigvals_flag=$evalf model.model_init=$init 
        #     python ../fit/fit_model.py latent=single_freq obs=gaussian model=fullrank latent.L=$L latent.seed=$seed \
        #         obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed model.model_init=$init 
        #     # python ../fit/fit_model.py latent=single_freq obs=gaussian model=fullrank_pinv latent.L=$L latent.seed=$seed \
        #     #     obs.ov1=$ov1 obs.ov2=$ov2 obs.seed=$seed model.model_init=$init 

