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

# python run_sgc_sep_est_bcn.py 1000 25 2 30 -3.5
# python plot_coh_sep_est_bcn.py 1000 25 2 30 -3.5
# python quick_test.py 1000 25 3 25 -3.5
# python alg_testing.py 1000 25 2 30 -3.5
# for c in 1 2 5 10 50 100 500
# for c in 25 50
# do 
#     for l in 5 25 50
#     do
#         for s in 1000
#         do
#             for mu in -6.0 -3.5 -1.0 
#             do
#                 python generate_synthetic_simple_mvcn.py $s $l 6 $c $mu
#                 python alg_testing.py $s $l 6 $c $mu
#             done
#         done
#     done
# done
