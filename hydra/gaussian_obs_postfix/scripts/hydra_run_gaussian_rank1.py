import os
import multiprocessing
import jax


from omegaconf import DictConfig
import hydra

from cohlib.jax.run_experiment_rank1temp import gen_data_and_fit_model_rank1
from cohlib.utils import pickle_save

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"
jax.config.update('jax_platform_name', 'cpu')
platform = jax.lib.xla_bridge.get_backend().platform.casefold()
print("Platform: ", platform)
print(len(jax.devices()))

@hydra.main(version_base=None, config_path="../../conf", config_name="config_gaussian_k3_rank1_nz9") # pylint: disable=no-member
def hydra_run(cfg: DictConfig) -> None:

    cfg.model.track_mus = True
    res = gen_data_and_fit_model_rank1(cfg)
    pickle_save(res, 'res.pickle')

if __name__ == "__main__":
    hydra_run() # pylint: disable=no-value-for-parameter