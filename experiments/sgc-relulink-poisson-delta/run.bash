win=1000
L=25
C=1
emiter=20
K=2
rho=0
kappa=0

# python generate_synthetic_simple_mvcn_nodc_relupoisson.py $win $L $K $C $alpha
# python alg_nodc_deltarelupoisson_refac.py $win $L $K $C $alpha $emiter $init_type
# python alg_nodc_deltarelupoisson_refac.py $win $L $K $C $alpha $emiter $rho $kappa

# alpha=500
# init_type=oracle
# optim_type=BFGS
# mu_only=True
# # python alg_nodc_deltarelupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
# python alg_nodc_deltarelupoisson_fixed_gamma_muonly.py $win $L $K $C $alpha $emiter $init_type $optim_type $mu_only

# optim_type=Newton
# for alpha in 200 500 1000
# do
alpha=50
for init_type in flat oracle
do
    for optim_type in BFGS Newton
    do
        python alg_nodc_deltarelupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
        python alg_nodc_deltaidpoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
    done
done
# for alpha in 50 100 150 200 250 
# for alpha in 300 350 400 450 500 
# do 
#     python alg_nodc_deltarelupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter
# done