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

# for alpha in 200 300 400 500 600
for alpha in 300
do
    python alg_delta_relupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
done



# for alpha in 200 500 1000
# do
# alpha=50
# for init_type in flat oracle
# do
#     for optim_type in BFGS Newton
#     do
#         python alg_nodc_deltarelupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#         python alg_nodc_deltaidpoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
#     done
# done
# for alpha in 50 100 150 200 250 
# for alpha in 300 350 400 450 500 
# do 
#     python alg_nodc_deltarelupoisson_fixed_gamma.py $win $L $K $C $alpha $emiter
# done