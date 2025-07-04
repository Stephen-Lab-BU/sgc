#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=02:15:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

gseed=0
evalf='fit'
init='empirical'
L=25
scalepower=50000
nfreqs=500
emiters=20
modT=10
nfreqsmodT=$((nfreqs * modT))
scalepowermodT=$((scalepower * modT))
nz_ind=14

for seed in {0..25}
do
    for K in 10
    do
        for mu in 2.3
        do
            python ../simulate/simulate_simple.py latent=single_freq_log obs=pp_log \
                latent.gamma_seed=$gseed latent.K=$K latent.L=$L latent.num_freqs=$nfreqsmodT latent.target_freq_ind=$nz_ind \
                latent.scale_power_target=$scalepowermodT latent.seed=$seed obs.mu=$mu obs.seed=$seed
            # python ../fit/fit_model.py latent=single_freq_log obs=pp_log model=fullrank latent.K=$K latent.L=$L latent.seed=$seed \
            #     obs.mu=$mu obs.seed=$seed model.model_init=$init 
            # PINV
            python ../fit/fit_model.py latent=single_freq_log obs=pp_log model=fullrank_pinv \
                latent.K=$K latent.L=$L latent.seed=$seed latent.num_freqs=$nfreqsmodT latent.target_freq_ind=$nz_ind \
                latent.scale_power_target=$scalepowermodT obs.mu=$mu obs.seed=$seed model.model_init=$init model.num_em_iters=$emiters
        done
    done
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

