L=50
C=1
# alpha=4
seed=8
K=2
win=1000
emiter=25

# python generate_synthetic_simple_mvcn_nodc_poisson_delta.py $win $L $K $C $alpha 
# python alg_nodc_poisson_delta_refac.py $win $L $K $C $alpha $emiters

# for alpha in 2.5 3.0 3.5 4.0 4.5
for alpha in 5.0 5.5 6.0 6.5 7.0 
do 
    python alg_nodc_poisson_delta_fixed_gamma.py $win $L $K $C $alpha $emiter
done
    
    