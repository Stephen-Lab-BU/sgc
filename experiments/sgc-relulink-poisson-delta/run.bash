win=1000
L=50
emiter=20
K=2
rho=0
kappa=0

alpha=400
Cdummy=1
# python generate_latent_deltarelu_single_freq.py $win $L $K $Cdummy $alpha 

python generate_latent_deltarelu.py $win $L $K $Cdummy $alpha 

# init_type=flat
# optim_type=Newton

# C=1
# alpha=400
# python alg_delta_relupoisson_single_freq_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type

# python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type

# for C in 1 2 5 10 20
# do 
#     python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
# done

# for init_type in flat oracle
# do
#     for optim_type in BFGS Newton
#     do
#         python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#     done
# done


# for init_type in flat oracle
# do
#     for optim_type in BFGS Newton
#     do
#         for alpha in 200 400 600 800 1000
#         do
#             python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#         done
#     done
# done

# alpha=400
# for init_type in flat oracle
# do
#     for optim_type in BFGS Newton
#     do
#         python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#         # python alg_delta_relupoisson_fixed_gamma_vmean.py $win $L $K $C $alpha $emiter $init_type $optim_type
#         # python alg_nodc_deltarelupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#         # python alg_nodc_deltaidpoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#     done
# done
