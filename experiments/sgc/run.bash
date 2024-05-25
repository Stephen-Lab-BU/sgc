
# python generate_synthetic_simple_mvcn_nodc.py 1000 25 2 25 -3.5 7
# python alg_nodc_fixed_gamma.py 1000 25 2 10 -2.5


# for c in 1 5 10 25 
# # for c in 250 500
# do 
#     # for mu in -5 -4 -3 -2 -1
# #     do 
#     for mu in 0 1 2 3 4 5
#     do 
#         python alg_nodc_fixed_gamma.py 1000 25 2 $c $mu
#     done
# done



L=50
C=25
alpha=-3.5

python generate_synthetic_simple_mvcn_nodc_bernoulli.py 1000 $L 2 $C $alpha 8
python alg_nodc_bernoulli.py 1000 $L 2 $C $alpha 10 


# next - calibrate and try same setup as for gaussian... sample length, trials and 3 mus 
# then run data creation over mus and look at xx xn nn - read lepage / aoi for further ideas 


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
