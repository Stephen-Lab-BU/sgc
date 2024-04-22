# python generate_synthetic_simple_mvcn_gaussian.py 1000 25 3 10 -1 5 -2
# python alg_gaussian_obs.py 1000 25 3 10 -1 5 -2


python alg_gaussian_obs_fixed_gamma.py 1000 25 3 10 -1 1 -2
# python generate_synthetic_simple_mvcn_gaussian.py 1000 25 3 15 -1 1.0 -2.0
# for c in 1 10 25 500
# do 
#     for o2 in 2 0 -2  
#     do
#         python alg_gaussian_obs_fixed_gamma_analytical.py 1000 25 3 $c -1 1 $o2
#     done
# done
