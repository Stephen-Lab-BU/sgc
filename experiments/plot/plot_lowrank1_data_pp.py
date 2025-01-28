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
from cohlib.utils import pickle_open

import cohlib.confs.utils as conf
from cohlib.confs.latent import BasicSingleFreqLog, BasicSingleFreqReLU
from cohlib.confs.obs import PPLogObs, PPReluObs
from cohlib.jax.dists import cif_alpha_log, cif_alpha_relu


jax_boilerplate()

@dataclass
class PlotParams:
    plot_type: str = 'data_plot'
    L: int = 50
    alpha: float = 1.0

defaults = [
    {"plot": "dataplot"},
    {"latent": "single_freq_log"},
    {"obs": "pp_log"},
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
cs.store(group='latent', name='single_freq_log', node=BasicSingleFreqLog)
cs.store(group='latent', name='single_freq_relu', node=BasicSingleFreqReLU)
cs.store(group='obs', name='pp_log', node=PPLogObs)
cs.store(group='obs', name='pp_relu', node=PPReluObs)
cs.store("config", node=Config)

@hydra.main(version_base=None, config_name = "config")
def plot(cfg: Config):
    os.chdir('/projectnb/stephenlab/jtauber/cohlib/experiments')
    cfg.latent.L = cfg.plot.L
    alpha = cfg.plot.alpha
    cfg.obs.alpha = alpha
    alphas = jnp.ones(cfg.latent.K)*alpha

    print(f'gathering data for alpha={alpha}')
    cfg.obs.alpha = alpha

    # load zs data and compute oracle est
    lcfg = cfg.latent
    latent_dir = conf.get_latent_dir(lcfg)
    latent_load = pickle_open(os.path.join(latent_dir, 'latent_sim.pkl'))
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
    obs_type = ocfg.obs_type
    obs_dir = conf.get_obs_dir(ocfg, latent_dir)
    obs_load = pickle_open(os.path.join(obs_dir, 'obs_sim.pkl'))
    obs = obs_load['obs']

    if obs_type == 'pp_log':
        lams = cif_alpha_log(alphas, xs)
    elif obs_type == 'pp_relu':
        lams = cif_alpha_relu(alphas, xs)
    else:
        raise ValueError

    plot_dir = get_plot_dir(cfg)
    if not os.path.exists(plot_dir):
        path = pathlib.Path(plot_dir)
        path.mkdir(parents=True, exist_ok=False)

    print(f'drawing and saving plots')
    Ks_plot = jnp.array([0])
    xs_plot= xs[:,Ks_plot,:]
    lams_plot= lams[:,Ks_plot,:]
    # obs_plot = obs[:,Ks_plot,:]

    fig,ax = plt.subplots(2,1,figsize=(6,4), sharex=True)
    for trial in range(lcfg.L):
        plot_synthetic_data_trial(ax[0], xs_plot, trial, 
        xylabs=['', 'latent'], title='xs')
        plot_synthetic_data_trial(ax[1], lams_plot, trial, 
        xylabs=['Time (sec)', 'rate'], title='lams')
    # ax.set_ylim([0,200])
    # ax.set_title(r'obs var=' f'{ocfg.ov1}e{ocfg.ov2}; ' + '$x^{0,\ell}$')
    ax[0].set_title(f'xs')
    if obs_type == 'pp_relu':
        ax[1].set_title('ReLU Link; ' + r'$\alpha$' + f' = {alpha}')
    elif obs_type == 'pp_log':
        ax[1].set_title('Log Link; ' + r'$\alpha$' + f' = {alpha}')
    else:
        raise ValueError
    plt.tight_layout()

    savename = f'plot-allxs-lseed{cfg.latent.seed}-oseed{cfg.obs.seed}.png'
    savepath = os.path.join(plot_dir,savename)
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

        plot_synthetic_data_trial(ax[i,0], lams, trial, color=colors, xylabs=[xlab, '$\\lambda^{k,\ell}$'])
        plot_synthetic_data_trial(ax[i,1], obs, trial, color=colors, xylabs=[xlab, '$n^{k,\ell}$'])
    plt.tight_layout()
    savename = f'plot-trials-xs-obs-lseed{cfg.latent.seed}-oseed{cfg.obs.seed}.png'
    savepath = os.path.join(plot_dir,savename)
    print(savepath)
    plt.savefig(savepath)


def get_plot_dir(cfg):
    ocfg = cfg.obs
    lcfg = cfg.latent
    pcfg = cfg.plot
    plot_dir = f'data/figs-data/{pcfg.plot_type}/latent-{lcfg.latent_type}/window-{int(2*lcfg.num_freqs)}/K{lcfg.K}/obs-{ocfg.obs_type}/alpha-{ocfg.alpha}'
    return plot_dir

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
