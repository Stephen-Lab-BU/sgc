win=1000
L=50
emiter=20
K=2
rho=0
kappa=0
seed=7

# do
#     python generate_latent_deltarelu.py $win $L $K $seed 
# done
# python generate_latent_single_freq_deltarelu.py $win $L $K $seed


init_type=oracle
optim_type=Newton
C=1
alpha=400

for init_type in flat oracle
do
    for optim_type in BFGS Newton
    do
        python alg_delta_relupoisson_fixed_gamma_xdatasinglefreq.py $win $L $K $C $alpha $emiter $init_type $optim_type
        python alg_delta_relupoisson_single_freq_fixed_gamma_xdatasimple.py $win $L $K $C $alpha $emiter $init_type $optim_type
    done
done
# python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type -seed $seed
# python alg_delta_relupoisson_single_freq_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type

# python alg_delta_relupoisson_fixed_gamma_singlefreq_data.py $win $L $K $C $alpha $emiter $init_type $optim_type 
# python alg_delta_relupoisson_single_freq_fixed_gamma_simple_data.py $win $L $K $C $alpha $emiter $init_type $optim_type 

