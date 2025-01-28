import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, List
import glob as glob

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
import jax.numpy as jnp
import jax.random as jr

from cohlib.jax.dists import LowRankCCN, sample_obs
from cohlib.jax.utils import add0, jax_boilerplate
from cohlib.utils import pickle_save, pickle_open

from cohlib.confs.latent.simple import BasicSingleFreqReLU, create_lrccn_basic_rank1
from cohlib.confs.obs.pp_relu import PPReluObs
from cohlib.confs.utils import get_latent_dir, get_obs_dir
from cohlib.confs.config import get_sim_config, register_configs

jax_boilerplate()

# Create config 

register_configs()
SimConfig = get_sim_config()


@hydra.main(version_base=None, config_name = "config")
def run(cfg: SimConfig) -> None:
    print('Simulating from config:')
    print(OmegaConf.to_yaml(cfg))
    lcfg = cfg.latent
    ocfg = cfg.obs


    latent_dir = get_latent_dir(lcfg)
    obs_dir = get_obs_dir(ocfg, latent_dir)

    # Try loading latent according to lcfg; run simulation of no pre-existing
    try: 
        load_latent = pickle_open(os.path.join(latent_dir, 'latent_sim.pkl'))
        print('Loading pre-existing latent simulation.')
        zs_nz = load_latent['zs_nz']
        freqs = load_latent['freqs']
        nz = load_latent['nz']
        zs_existed = True
    except FileNotFoundError:
        print('Simulating Latent')
        if not os.path.exists(latent_dir):
            path = pathlib.Path(latent_dir)
            path.mkdir(parents=True, exist_ok=False)

        lrccn = create_lrccn_basic_rank1(lcfg)
        lrk_sample_seed = lcfg.seed #+ lcfg.L
        lrk_sample = jr.key(lrk_sample_seed)
        zs_nz = lrccn.sample_nz(lrk_sample, lcfg.L)
        freqs = lrccn.freqs
        nz = lrccn.nz
        lcfg_resolved = OmegaConf.to_container(lcfg, resolve=True)
        latent_save = {'lcfg': lcfg_resolved, 'zs_nz': zs_nz, 'nz': nz, 'freqs': freqs}
        latent_savename = os.path.join(latent_dir, 'latent_sim.pkl')
        pickle_save(latent_save, latent_savename) 
        zs_existed = False

    # zs
    num_freqs = freqs.size
    zs = jnp.zeros((num_freqs, lcfg.K, lcfg.L), dtype=complex)
    zs = zs.at[nz,:,:].set(zs_nz)
    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    
    try: 
        obs_save_path = os.path.join(obs_dir, 'obs_sim.pkl')
        obs_load = pickle_open(obs_save_path)
        if 'obs' in obs_load.keys():
            print('Observations already simulated and saved.')

    except FileNotFoundError:
        print('Simulating Observations')
        obs_params = ocfg
        obs = sample_obs(xs, obs_params)

        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        obs_save = {'cfg': cfg_resolved, 'obs': obs}
        if not os.path.exists(obs_dir):
            path = pathlib.Path(obs_dir)
            path.mkdir(parents=True, exist_ok=False)
        obs_save_path = os.path.join(obs_dir, 'obs_sim.pkl')
        pickle_save(obs_save, obs_save_path)

        # Save hydra run info 
        logging_data = {'cfg': cfg_resolved,
                        'zs_existed': zs_existed, 
                        'obs_loc': obs_dir,
                        'latent_loc': latent_dir}

        pickle_save(logging_data, os.path.join(obs_dir, 'log.pkl'))

    # TODO plot / save image


if __name__ == "__main__":
    run()
