import time
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, List
import glob as glob

import hydra
from hydra import compose
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cohlib.jax.utils import jax_boilerplate
from cohlib.jax.dists import naive_estimator
from cohlib.utils import pickle_save, pickle_open

import cohlib.confs.utils as conf
from cohlib.confs.latent.simple import BasicSingleFreq, BasicSingleFreqReLU, BasicSingleFreqLog
from cohlib.confs.obs import GaussianObs, PPReluObs, PPLogObs
from cohlib.confs.model.simple_lcfg_inherit import FullRankToySimple
from cohlib.jax.plot import get_eigval, get_eigvec


jax_boilerplate()

def get_eigvecs(ccn):
    """
    ccn.gamma shape (J x K x K)
    """
    gamma = ccn.gamma
    _, eigvecs_raw = jnp.linalg.eigh(gamma)
    eigvecs = eigvecs_raw[:,:,::-1]
    return eigvecs

def rotate_eigvecs(eigvecs):
    """
    Args:
        eigvecs: J x K x R
    """

    J = eigvecs.shape[0]
    R = eigvecs.shape[2]

    rotated = jnp.zeros_like(eigvecs)

    for j in range(J):
        thetas = jnp.angle(eigvecs[j,0,:])
        rotations = jnp.exp(-1j*thetas)

        rotated = rotated.at[j,:,:].set(eigvecs[j,:,:] * rotations[None,:])
    
    return rotated

def mod_config(cfg, L, theta):
    cfg.latent.L = L
    if cfg.obs.obs_type == 'gaussian':
        cfg.obs.ov2 = theta
    elif cfg.obs.obs_type in ['pp_relu', 'pp_log']:
        cfg.obs.mu = theta
    else:
        raise ValueError

    return cfg

def get_theta_label(ocfg, theta):
    if ocfg.obs_type == 'gaussian':
        label = f'{ocfg.ov1}e{ocfg.ov2}'
    elif ocfg.obs_type in ['pp_relu', 'pp_log']:
        label = f'{ocfg.mu}'
    else:
        return ValueError
    return label

@dataclass
class PlotParams:
    plot_type: str = 'eigvec_var_L'
    Ls: List[int] = field(default_factory=lambda: [50, 100])
    thetas: List[float] = field(default_factory=lambda: [25, 50])
    rank_plot: int = 1
    rotate: bool = True
    dims: List[int] = field(default_factory=lambda: [0,1,2])

defaults = [
    {"plot": "lowrank"},
    {"latent": "single_freq_log"},
    {"obs": "pp_relu"},
    {"model": "fullrank"}
]

@dataclass 
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    plot: Any = MISSING
    latent: Any = MISSING
    obs: Any = MISSING
    model: Any = MISSING

cs = ConfigStore.instance()
cs.store(group='plot', name='lowrank', node=PlotParams)
cs.store(group='latent', name='single_freq', node=BasicSingleFreq)
cs.store(group='latent', name='single_freq_relu', node=BasicSingleFreqReLU)
cs.store(group='latent', name='single_freq_log', node=BasicSingleFreqLog)
cs.store(group='obs', name='gaussian', node=GaussianObs)
cs.store(group='obs', name='pp_relu', node=PPReluObs)
cs.store(group='obs', name='pp_log', node=PPLogObs)
cs.store(group='model', name='fullrank', node=FullRankToySimple)
cs.store("config", node=Config)



@hydra.main(version_base=None, config_name = "config")
def plot(cfg: Config):
    os.chdir('/projectnb/stephenlab/jtauber/cohlib/experiments')
    Ls = list(cfg.plot.Ls)
    thetas = list(cfg.plot.thetas)
    eigrank = cfg.plot.rank_plot

    plot_data = {}

    for l, L in enumerate(Ls):
        for t, theta in enumerate(thetas):
            print(f'gathering data for L={L}, theta={theta}')
            cfg = mod_config(cfg, L, theta)

            # context initialization
            nz_model = jnp.array([cfg.latent.target_freq_ind])
            temp_params = {'lcfg': cfg.latent, 'ocfg': cfg.obs, 
                           'nz_model': nz_model, 'K': cfg.latent.K}
            model_dir = conf.get_model_dir(cfg, temp_params)

            res = pickle_open(os.path.join(model_dir, 'res.pkl'))

            # load result value
            # load true value
            lrccn_true = conf.create_lrccn_basic_rank1(cfg.latent)
            plot_data[t,l] = {}
            plot_data[t,l]['eigvec_true'] = lrccn_true.eigvecs[0,:,eigrank-1]
            if cfg.plot.rotate is True: 
                eigvecs_em = jnp.stack([rotate_eigvecs(get_eigvecs(x))[0,:,eigrank-1] for x in res['track']['ccn']])
            else:
                eigvecs_em = jnp.stack([get_eigvecs(x)[0,:,eigrank-1] for x in res['track']['ccn']])
            plot_data[t,l]['eigvecs_em'] = eigvecs_em

            # load zs data and compute oracle est
            latent_dir = conf.get_latent_dir(cfg.latent)
            latent_load = pickle_open(os.path.join(latent_dir, 'latent_sim.pkl'))
            zs_nz = latent_load['zs_nz']
            nz_true = latent_load['nz']
            jind_nz = 0

            # TODO fix for 
            gamma_oracle = jnp.einsum('jkl,jil->jkil', zs_nz, zs_nz.conj()).mean(-1)
            oracle_eigvec = get_eigvec(gamma_oracle[jind_nz,:,:], eigrank)
            oracle_eigvec = oracle_eigvec*jnp.exp(-1j*jnp.angle(oracle_eigvec[0]))
            plot_data[t,l]['eigvec_oracle'] = oracle_eigvec

            obs_dir = conf.get_obs_dir(cfg.obs, latent_dir)
            obs_load = pickle_open(os.path.join(obs_dir, 'obs_sim.pkl'))
            obs = obs_load['obs']

            if cfg.model.inherit_lcfg is True:
                nz_model = nz_true
            else:
                raise NotImplementedError

            obs_type = cfg.obs.obs_type
            if obs_type == 'gaussian':
                naive_est = naive_estimator(obs, nz_model)
            elif obs_type in ['pp_relu', 'pp_log']:
                naive_est = naive_estimator(obs, nz_model)*1e6
            else:
                raise ValueError
            naive_eigvec = get_eigvec(naive_est[jind_nz,:,:], eigrank)
            naive_eigvec = naive_eigvec*jnp.exp(-1j*jnp.angle(naive_eigvec[0]))
            plot_data[t,l]['eigvec_naive'] = naive_eigvec

    print(f'drawing and saving plots')
    dims = cfg.plot.dims
    funcs = [jnp.angle, jnp.abs, jnp.real, jnp.imag]
    funcnames = ['Phase', 'Mag', 'Real', 'Imag']
    for func, funcname in zip(funcs, funcnames):
        for dim in dims:

            fig, ax = plt.subplots(len(thetas), len(Ls), figsize=(12,8), sharex=True, sharey='row')
            for l, L in enumerate(Ls):
                for t, theta in enumerate(thetas):
                    cfg = mod_config(cfg, L, theta)

                    plot_eigvec_var_L_subplot(ax[t,l], plot_data[t,l], dim, func, funcname)

                    if l == 0:
                        ax[t,l].set_ylabel(f'{get_theta_label(cfg.obs, theta)}')
                    if t == 0:
                        ax[t,l].set_title(f'L = {L}')
                    if t == len(thetas)-1:
                        ax[t,l].set_xlabel('EM Iter')
                    if funcname == 'Phase':
                        ax[t,l].set_ylim([-jnp.pi,jnp.pi])

            fig.suptitle(f'{nz_model[jind_nz]+1} Hz ' + rf'Eigvec({eigrank}) dim={dim+1} {funcname}')
            plt.tight_layout()
            plot_dir = get_plot_dir(cfg, dim, funcname)
            if not os.path.exists(plot_dir):
                path = pathlib.Path(plot_dir)
                path.mkdir(parents=True, exist_ok=False)
            savename = f'plot-lseed{cfg.latent.seed}-oseed{cfg.obs.seed}.png'
            savepath = os.path.join(plot_dir,savename)
            print(savepath)
            plt.savefig(savepath)

def get_plot_dir(cfg, dim, funcname):
    plot_dir = f'data/figs-fit/{cfg.plot.plot_type}/latent-{cfg.latent.latent_type}/fullrank/window-{int(2*cfg.latent.num_freqs)}/K{cfg.latent.K}/obs-{cfg.obs.obs_type}/{cfg.model.model_init}/func-{funcname}/dim-{dim}'
    return plot_dir

def plot_eigvec_var_L_subplot(ax, plot_dict, dim, func, funcname):
    naive_color = 'tab:red'
    naive_style = 'dashed'
    naive_width = 2

    true_color = 'k'
    true_style = 'solid'
    true_width = 2

    oracle_color = 'tab:green'
    oracle_style = 'dotted'
    oracle_width = 2.5

    est_color = 'tab:blue'
    est_style = 'solid'
    est_width = 2

    ev_true = plot_dict['eigvec_true']
    ev_oracle = plot_dict['eigvec_oracle']
    ev_naive = plot_dict['eigvec_naive']
    evs_em = plot_dict['eigvecs_em']

    func_eigvecs_em = jnp.stack([func(x[dim]) for x in evs_em])

    ax.axhline(func(ev_true[dim]), linestyle=true_style, color=true_color, linewidth=true_width)
    ax.axhline(func(ev_oracle[dim]), linestyle=oracle_style, color=oracle_color, linewidth=oracle_width)
    ax.axhline(func(ev_naive[dim]), linestyle=naive_style, color=naive_color, linewidth=naive_width)
    ax.plot(func_eigvecs_em, color=est_color, linestyle=est_style, linewidth=est_width)



if __name__ == "__main__":
    plot()
