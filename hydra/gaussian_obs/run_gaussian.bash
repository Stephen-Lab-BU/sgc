for init in flat-init
do
    for L in 25 #50
    do 
        for ov2 in -3 -2 -1 0
        do
            python hydra_run_gaussian.py latent.L=$L 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
            hydra.job.chdir=True
            for method in old oldmod # jax
            do
                python hydra_run_gaussian_ts.py latent.L=$L model.ts_method=$method 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
                hydra.job.chdir=True
            done
            for ic in true false
            do
                python hydra_run_gaussian_scipy.py latent.L=$L model.inverse_correction=$ic 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
                hydra.job.chdir=True
            done
        done
    done
done

# for ic in true false
# do
#     for init in flat-init
#     do
#         for L in 25 #50
#         do 
#             for ov2 in -3 -2 -1 0
#             do
#                 python hydra_run_gaussian.py latent.L=$L 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
#                 hydra.job.chdir=True
#                 # python hydra_run_gaussian_ts.py latent.L=$L 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
#                 # hydra.job.chdir=True
#                 python hydra_run_gaussian_scipy.py latent.L=$L model.inverse_correction=$ic 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
#                 hydra.job.chdir=True
#             done
#         done
#     done
# done

# # python hydra_run_gaussian.py obs=test model=test hydra.job.chdir=True 
# # python hydra_run_gaussian.py hydra.job.chdir=True 
