
# python generate_synthetic_simple_mvcn_nodc_poisson.py 1000 25 2 1 -3.5 7
# python alg_nodc_fixed_gamma_poisson.py 1000 25 2 25 -2.0


# for c in 1 5 10 25 
# # for c in 250 500
# # for c in 1
# do 
#     # for mu in -5 -4 -3 -2 -1
#     for mu in 6 7 8
#     do 
#         python alg_nodc_fixed_gamma_poisson.py 1000 25 2 $c $mu
#     done
# done

# python generate_synthetic_simple_mvcn_nodc_relupoisson.py 1000 50 2 25 0.2
# python alg_nodc_fixed_gamma_relupoisson_refac.py 1000 25 2 25 0.1 10

# python generate_synthetic_simple_mvcn_nodc_relupoisson.py 1000 25 2 25 0.1
# python alg_nodc_relupoisson_refac.py 1000 25 2 25 0.1 10 100 1


# python generate_synthetic_simple_mvcn_nodc_relupoisson.py 1000 50 2 25 0.1
# python alg_nodc_relupoisson_refac.py 1000 50 2 25 0.2 10 100 1

win=1000
L=25
C=1
emiter=25
alpha=2.5
K=2

python generate_synthetic_simple_mvcn_nodc_idpoisson.py $win $L $K $C $alpha
python alg_nodc_idpoisson_refac.py $win $L $K $C $alpha $emiter