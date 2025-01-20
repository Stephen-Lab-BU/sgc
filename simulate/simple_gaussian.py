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

from cohlib.jax.dists import LowRankCCN, add
from cohlib.utils import pickle_save, pickle_open

@dataclass
class BasicSingleFreq:
    latent_type: str = 'basic_single'
    K: int = 3
    num_freqs: int = 500 # assuming 1 second window 
    # target_freq_inds: List[int] = field(default_factory=lambda: [9])
    target_freq_ind: int = 9
    scale_power_target: float = 1.0e3
    L: int = 50
    seed: int = 7 

@dataclass
class GaussianObs:
    obs_type: str = 'gaussian'
    ov1: int = 1
    ov2: int = -3
    seed: int = 7
    

defaults = [
    {"latent": "singlefreq"},
    {"observation": "gaussian"}
]

@dataclass 
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    latent: Any = MISSING
    observation: Any = MISSING

cs = ConfigStore.instance()
cs.store(group='latent', name='singlefreq', node=BasicSingleFreq)
cs.store(group='observation', name='gaussian', node=GaussianObs)
cs.store("config", node=Config )


@hydra.main(version_base=None, config_name = "config")
def run(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    lcfg = cfg.latent
    ocfg = cfg.observation
    window = int(2*lcfg.num_freqs)
    print('test')
    print(OmegaConf.to_yaml(lcfg))

    # paths? 
    # Data/LatentType/Window/K/L/LatentSeed/ObsType/ObsParams/ObsSeed/
    # Data/LatentType/Window/K/L/LatentSeed contains pkl with lrccn and zs
    # Data/LatentType/Window/K/L/LatentSeed/ObsType/ObsParams/ObsSeed/ contains pkl with observations and summary fig
    latent_path = f'experiments/{lcfg.latent_type}/{window}/{lcfg.K}/{lcfg.L}/{lcfg.seed}'
    obs_subpath = f'{ocfg.obs_type}/{ocfg.ov1}-{ocfg.ov2}/{ocfg.seed}'
    obs_path = os.path.join(latent_path, obs_subpath)


    # lrccn_exists = True
    # contents = [os.path.split(x)[0] for x in glob.glob(os.path.join(latent_path), '*')]
    # if lrccn_exists is False:

    # if zs / 
    try: 
        load_latent = pickle_open(os.path.join(latent_path, 'latent_sim.pkl'))
        zs = load_latent['zs']
    except FileNotFoundError:
        if os.path.exists(latent_path):
            path = pathlib.Path(latent_path)
            path.mkdir(parents=True, exist_ok=False)

        lrccn = create_lrccn_basic_rank1(lcfg)
        lrk = lcfg.seed + lcfg.L
        zs = lrccn.sample_nz(lrk, lcfg.L)
        latent_save = {'lcfg': lcfg, 'zs': zs}
        latent_savename = os.path.join(latent_path, 'latent_sim.pkl')
        pickle_save(latent_save, latent_savename) 

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    
    obs_params = ocfg
    obs = sample_obs(xs, obs_params)
    obs_type = ocfg.obs_type

    if not os.path.exists(obs_path):
        path = pathlib.Path(obs_path)
        path.mkdir(parents=True, exist_ok=False)



    # next - simualte and save observations 

    # and then - how will we run model??
    # subfolder of latent seed with 'fits' - model results saved there according to a 'model_path'

    # then have option of running generation separately or or in conjunction with model fitting
    # won't be wasting run time reproducing simulations



    # TODO save result of *hydra run*
    # bool: lrccn and zs created or not 
    # str: location of lrccn
    # str: location of obs

    # TODO save observations
    # TODO plot / save image



def create_lrccn_basic_rank1(lcfg):
    K = lcfg.L
    N = lcfg.num_freqs
    target_freq_ind = lcfg.target_freq_ind
    seed = lcfg.seed
    freqs = jnp.arange(N)

    scale_target = lcfg.scale_power_target

    nz = jnp.array([target_freq_ind])
    R = 1

    Nnz = nz.size

    eigvecs_target = jnp.zeros((Nnz,K,R), dtype=complex)
    eigvals_target = jnp.zeros((Nnz,R))

    lrk = jr.key(seed)
    lrksplit = jr.split(lrk, Nnz)
    for j in range(Nnz):
        jrk = lrksplit[j]
        jrksplit = jr.split(jrk, R)
        for r in range(R):
            rjrk = jrksplit[r]
            phases = jr.uniform(rjrk, (K,), minval=-jnp.pi, maxval=jnp.pi)
            reals = jnp.cos(phases)
            imags = jnp.sin(phases)

            eigvec = reals + 1j*imags
            eigvec = eigvec / jnp.linalg.norm(eigvec)
            eigvec = eigvec*jnp.exp(-1j*jnp.angle(eigvec[0]))

            eigvecs_target = eigvecs_target.at[j,:,r].set(eigvec)

            eigval = K*scale_target + j + r
            eigvals_target = eigvals_target.at[j,r].set(eigval)

    lrccn = LowRankCCN(eigvals_target, eigvecs_target, K, freqs, nz)
    return lrccn

    # move ipynb code here to run as script
    # save dist object as output to be loaded in experiments
    # run plotting code and save figure showing overview 



if __name__ == "__main__":
    run()
