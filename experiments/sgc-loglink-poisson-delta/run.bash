L=50
C=1
# alpha=3.5
seed=8
K=2
win=1000
emiter=25
# python generate_synthetic_logpoisson.py $win $L $K $C $alpha 



# init_type="flat"
# optim_type="BFGS"
# for alpha in 5.3 5.7 6 6.2 6.4
# do
#     python alg_delta_logpoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
# done
alpha=4.0
for init_type in flat oracle
do
    for optim_type in BFGS Newton
    do
        python alg_delta_logpoisson_fixed_gamma.py $win $L $K $C $alpha $emiter $init_type $optim_type
    done
done

    
    