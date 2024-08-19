
for init in true-init 
do
    for L in 1000
    do 
        for ov2 in 0 3
        do
            python hydra_run.py latent.L=$L 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
            hydra.job.chdir=True
            python hydra_run.py latent.L=$L model.init=$init model.emiters=20 obs.ov2=$ov2 \
            hydra.job.chdir=True
        done
    done
done


for init in flat-init
do
    for L in 10 50 100 1000
    do 
        for ov2 in -5 -3 0 3
        do
            python hydra_run.py latent.L=$L 'model.support=[0,50]' model.emiters=20 model.init=$init obs.ov2=$ov2 \
            hydra.job.chdir=True
            python hydra_run.py latent.L=$L model.init=$init model.emiters=20 obs.ov2=$ov2 \
            hydra.job.chdir=True
        done
    done
done