for init in flat-init
do
    for L in 50
    do 
        for ov2 in -1 0 1 2
        # for ov2 in 1
        do
            for scale_init in 10 
            # for scale_init in 10 1000 100000
            do
                for ic in true 
                do
                    python hydra_run_gaussian_scipy.py latent.L=$L model.inverse_correction=$ic 'model.support=[0,50]' model.emiters=20 model.init=$init model.scale_init=$scale_init obs.ov2=$ov2 \
                    hydra.job.chdir=True
                done
            done
        done
    done
done

# for init in flat-init
# do
#     for L in 50
#     do 
#         for ov2 in -1 1 
#         do
#             for scale_init in 10 1000 100000
#             do
#                 for method in jax 
#                 do
#                     python hydra_run_gaussian_ts.py latent.L=$L model.ts_method=$method 'model.support=[0,50]' model.emiters=20 model.init=$init model.scale_init=$scale_init obs.ov2=$ov2 \
#                     hydra.job.chdir=True
#                 done
#             done
#         done
#     done
# done
