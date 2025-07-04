#!/bin/bash -l

#$ -pe omp 28
#$ -l h_rt=00:10:00

# When a parallel environment is requested the environment variable NSLOTS is set to the number of cores requested. This variable can be used within a program to setup an appropriate number of threads or processors to use.
# For example, some programs rely on the environment variable OMP_NUM_THREADS for parallelization:
OMP_NUM_THREADS=$NSLOTS
conda activate sgc_env

evalf='fit'
init='empirical'
L=50
K=25
nfreqs=500
emiters=20
modT=10
nfreqsmodT=$((nfreqs * modT))
nz_ind=14
exp_type='bottomK'

python ../fit/fit_model_app.py app.exp_type=$exp_type latent=app_fullrank_single_freq obs=app_pp_log model=fullrank_pinv \
    latent.K=$K latent.L=$L latent.num_freqs=$nfreqsmodT latent.target_freq_ind=$nz_ind \
    model.model_init=$init model.num_em_iters=$emiters model.m_step_option=testing-factor-igprior
