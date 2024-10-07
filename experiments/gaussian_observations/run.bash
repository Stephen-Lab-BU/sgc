# for c in 1 10 25 500
# do
#     for o2 in 2 0 -2
#     do 
#         python alg_gaussian_obs_fixed_gamma_analytical.py 1000 25 2 $c 0.0 1.0 $o2 10
#     done
# done

# for o1 in 10 9 8 # 7 6 5 4 3 2 1
for o1 in 1 
do 
    for o2 in -3 -2 -1 0
    do 
        # python jax_comp_alg_gaussian_obs_fixed_gamma_scale_hess_mod.py 1000 25 2 1 0.0 $o1 $o2 20
        python jax_comp_alg_gaussian_obs_fixed_gamma_scale_mod_test_correction.py 1000 25 2 1 0.0 $o1 $o2 20
        # python alg_gaussian_obs_fixed_gamma_confirm.py 1000 25 2 1 0.0 $o1 $o2 20
    done
done

# python generate_synthetic_simple_mvcn_gaussian_nodc.py 1000 25 2 1 0.0 1.0 0.0 7


# for l in 10 100 500
#     for c in 1 10
#     do 
#         for o2 in 2 0 -2  
#         do
#             python alg_gaussian_obs_fixed_gamma_analytical.py 1000 25 3 $c -1 1 $o2
#         done
#     done
