win=1000
L=50
C=1
emiter=20
K=2
rho=0
kappa=0

# python generate_synthetic_relupoisson.py $win $L $K $C $alpha 

init_type=flat
optim_type=BFGS
# alpha=350.0

for alpha in 200 400 600 800 1000
# for alpha in 400
do
    python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#     python alg_delta_relupoisson_fixed_gamma_vmean.py $win $L $K $C $alpha $emiter $init_type $optim_type
done

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
