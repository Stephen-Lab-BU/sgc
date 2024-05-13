
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

python generate_synthetic_simple_mvcn_nodc_relupoisson.py 1000 25 2 25 0.1
