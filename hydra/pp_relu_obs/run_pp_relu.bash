emiters=20

for L in 50 #100 250 
do 
    # for alpha in 0 50 100 300
    for alpha in 200 
    do 
        python hydra_run_pp_relu.py latent.L=$L 'model.support=[0,50]' model.emiters=$emiters obs.alpha=$alpha \
        hydra.job.chdir=True
    done
done
