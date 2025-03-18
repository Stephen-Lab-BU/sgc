import os
import multiprocessing
import jax
import jax.numpy as jnp


from hydra import initialize, compose 
# from omegaconf import DictConfig
# import hydra

from cohlib.jax.run_experiment_rankRmodel import gen_data_and_fit_model_rankRm
from cohlib.utils import pickle_open

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"
jax.config.update('jax_platform_name', 'cpu')
platform = jax.lib.xla_bridge.get_backend().platform.casefold()
print("Platform: ", platform)
print(len(jax.devices()))

def test_hydra_run_refac():

    with initialize(version_base=None, config_path="../hydra/conf"):
        cfg = compose(config_name="config_gaussian_k3_rank1_nz9")


        K = 3
        seed = 8
        cfg.model.track_mus = True
        cfg.obs.ov1 = 5.0
        cfg.obs.ov2 = -1.0 
        cfg.obs.seed = seed
        cfg.model.scale_init = 100
        cfg.model.init = 'oracle-init'
        cfg.model.emiters = 5
        cfg.model.support = [9,9]
        cfg.model.maxiter = 10
        cfg.latent.L = 25
        cfg.latent.seed = seed
        cfg.latent.gamma = f'k{K}-chlg4-rotate-gaussian-rank1-nz9'

        cfg.model.ts_flag = 'fixed_eigval_true'
        cfg.model.m_step_init=False
        cfg.model.ts_flag2 = 'eigh_est'
        refac_res = gen_data_and_fit_model_rankRm(cfg)

    saved_res = pickle_open('/projectnb/stephenlab/jtauber/cohlib/tests/outputs/2025-01-10/09-21-28/res.pickle')

    refac_num_iters = len(refac_res['track']['gamma_lowrank'])

    refac_eigvec_track = jnp.stack([refac_res['track']['gamma_lowrank'][i].eigvecs.squeeze() for i in range(refac_num_iters)])
    saved_eigvec_track = jnp.stack([saved_res['track']['gamma_lowrank'][i].eigvecs.squeeze() for i in range(refac_num_iters)])

    assert jnp.all(jnp.isclose(refac_eigvec_track, saved_eigvec_track, 1e-6))
