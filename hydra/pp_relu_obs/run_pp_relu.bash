
for init in true-init
do
    for L in 50
    do 
        # for alpha in 0 50 100 300
        for alpha in 300
        do 
            # for init_mod in 0.01 0.001 0.0001
            for init_mod in 0.001
            do
                python hydra_run_pp_relu.py latent.L=$L model.init_mod=$init_mod 'model.support=[0,50]' model.emiters=20 model.init=$init obs.alpha=$alpha \
                hydra.job.chdir=True
            done
        done
    done
done
