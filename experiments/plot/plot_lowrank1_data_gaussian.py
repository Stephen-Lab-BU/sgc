import time
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, List
import glob as glob

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import jax.numpy as jnp
import numpy
import matplotlib.pyplot as plt

from cohlib.jax.utils import jax_boilerplate, add0
from cohlib.jax.dists import naive_estimator
from cohlib.utils import pickle_save, pickle_open

import cohlib.confs.utils as conf
from cohlib.confs.latent.simple import BasicSingleFreq
from cohlib.confs.obs.gaussian import GaussianObs
from cohlib.confs.model.simple_lcfg_inherit import LowRankToySimpleM1
from cohlib.jax.plot import get_eigval, get_eigvec


jax_boilerplate()

@dataclass
class PlotParams:
    plot_type: str = 'data_plot'
    L: int = 25
    ov2: float = -1.0

defaults = [
    {"plot": "dataplot"},
    {"latent": "single-freq"},
    {"obs": "gaussian"},
]

@dataclass 
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    plot: Any = MISSING
    latent: Any = MISSING
    obs: Any = MISSING
    model: Any = MISSING

cs = ConfigStore.instance()
cs.store(group='plot', name='dataplot', node=PlotParams)
cs.store(group='latent', name='single-freq', node=BasicSingleFreq)
cs.store(group='obs', name='gaussian', node=GaussianObs)
cs.store("config", node=Config)

@hydra.main(version_base=None, config_name = "config")
def plot(cfg: Config):
    os.chdir('/projectnb/stephenlab/jtauber/cohlib/experiments')
    cfg.latent.L = cfg.plot.L
    ov2 = cfg.plot.ov2
    cfg.obs.ov2 = ov2

    plot_data = {}

    print(f'gathering data for  ov2={ov2}')
    cfg.obs.ov2 = ov2

    # load zs data and compute oracle est
    lcfg = cfg.latent
    latent_path = conf.get_latent_path(lcfg)
    latent_load = pickle_open(os.path.join(latent_path, 'latent_sim.pkl'))
    zs_nz = latent_load['zs_nz']
    nz = latent_load['nz']
    freqs = latent_load['freqs']
    jind_nz = 0
    num_freqs = freqs.size

    zs = jnp.zeros((num_freqs, lcfg.K, lcfg.L), dtype=complex)
    zs = zs.at[nz,:,:].set(zs_nz)
    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    ocfg = cfg.obs
    obs_path = conf.get_obs_path(ocfg, latent_path)
    obs_load = pickle_open(os.path.join(obs_path, 'obs_sim.pkl'))
    obs = obs_load['obs']
    obs_type = ocfg.obs_type

    plot_path = get_plot_path(cfg)
    if not os.path.exists(plot_path):
        path = pathlib.Path(plot_path)
        path.mkdir(parents=True, exist_ok=False)

    print(f'drawing and saving plots')
    Ks_plot = jnp.array([0])
    xs_plot= xs[:,Ks_plot,:]
    # obs_plot = obs[:,Ks_plot,:]

    fig,ax = plt.subplots(figsize=(6,2))
    for trial in range(lcfg.L):
        plot_synthetic_data_trial(ax, xs_plot, trial, 
        xylabs=['Time (sec)', 'rate'])
    # ax.set_ylim([0,200])
    # ax.set_title(r'obs var=' f'{ocfg.ov1}e{ocfg.ov2}; ' + '$x^{0,\ell}$')
    ax.set_title(f'xs')

    savename = f'plot-allxs-lseed{cfg.latent.seed}-oseed{cfg.obs.seed}.png'
    savepath = os.path.join(plot_path,savename)
    print(savepath)
    plt.savefig(savepath)

    # Ks_plot = jnp.array([0,5,10,15])
    fig,ax = plt.subplots(3,2,figsize=(10,5), sharex=True)
    trials = jnp.arange(3)+1

    colors = plt.cm.cividis(jnp.linspace(0,1,lcfg.K))
    for i, trial in enumerate(trials):
        if trial == trials[-1]:
            xlab='Time (sec)'
        else:
            xlab=''

        plot_synthetic_data_trial(ax[i,0], xs, trial, color=colors, xylabs=[xlab, '$\\lambda^{k,\ell}$'])
        plot_synthetic_data_trial(ax[i,1], obs, trial, color=colors, xylabs=[xlab, '$n^{k,\ell}$'])
    plt.tight_layout()
    savename = f'plot-trials-xs-obs-lseed{cfg.latent.seed}-oseed{cfg.obs.seed}.png'
    savepath = os.path.join(plot_path,savename)
    print(savepath)
    plt.savefig(savepath)


def get_plot_path(cfg):
    ocfg = cfg.obs
    lcfg = cfg.latent
    pcfg = cfg.plot
    plot_path = f'data/figs-data/{pcfg.plot_type}/latent-{lcfg.latent_type}/window-{int(2*lcfg.num_freqs)}/K{lcfg.K}/obs-{ocfg.obs_type}/ovb{ocfg.ov1}-ove{ocfg.ov2}'
    return plot_path

def plot_synthetic_data_trial(ax, data, trial, title=None, xylabs=None, color='tab:blue'):
    title_size = 12
    label_size = 10
    x = jnp.arange(0, 1000) / 1000
    # for i in range(start,start+3):
    l = trial
    K = data.shape[1]
    # i = 3
        
    for k in range(K):
        if type(color) is str:
            ax.plot(x, data[:,k,l], color=color)
        elif type(color) is numpy.ndarray:
            ax.plot(x, data[:,k,l], color=color[k])
        else:
            raise ValueError
    if title is None:
        ax.set_title(f'Trial {trial}', size=title_size)
    ax.margins(0)

    if xylabs is None:
        ax.set_ylabel('Intensity', size = label_size)
        ax.set_xlabel('Time (sec)', size = label_size)
    elif len(xylabs) == 2:
        ax.set_xlabel(xylabs[0], size = label_size)
        ax.set_ylabel(xylabs[1], size = label_size)
    else:
        raise ValueError
    # plt.xlim([0,sample_length/fs])



if __name__ == "__main__":
    plot()
