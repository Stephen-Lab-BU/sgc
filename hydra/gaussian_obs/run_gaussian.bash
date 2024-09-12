# for init in flat-init
# do
#     for L in 50
#     do 
#         for ov2 in -3 0
#         do
#             python hydra_run_gaussian.py latent.L=$L 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
#             hydra.job.chdir=True
#         done
#     done
# done

# python hydra_run_gaussian.py obs=test model=test hydra.job.chdir=True 
python hydra_run_gaussian.py hydra.job.chdir=True 
