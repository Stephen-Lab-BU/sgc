import os
from pathlib import Path
from omegaconf import OmegaConf
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from cohlib.utils import  pickle_open



import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_eigvals_em_iters(ax, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None):
    eigs = jnp.array([jnp.linalg.eigh(gamma_init[j_ind_full,:,:])[0]] + [jnp.linalg.eigh(gamma_r[j_ind,:,:])[0]  for gamma_r in gamma_iter_list])
    color = plt.cm.rainbow(jnp.linspace(0, 1, len(eigs)))
    for i, e in enumerate(eigs):
        ax.plot(e[::-1], color=color[i], linewidth=1)

def get_eigval(mat, rank):
    eigvals = jnp.linalg.eigh(mat)[0]
    return eigvals[-rank]

def get_eigvec(mat, rank):
    eigvecs = jnp.linalg.eigh(mat)[1]
    return eigvecs[:,-rank]

def plot_cross_spec_eigval_em_iters(ax, eigrank, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None, 
        color='tab:blue', style='-', width=2):
    eigs = jnp.array([get_eigval(gamma_init[j_ind_full,:,:], eigrank)] + [get_eigval(gamma_r[j_ind,:,:], eigrank) for gamma_r in gamma_iter_list])
    ax.plot(eigs, color=color, linewidth=width, linestyle=style)

def get_eigvec_em_iters(eigrank, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None):
    eigvecs = jnp.array([get_eigvec(gamma_init[j_ind_full,:,:], eigrank)] + [get_eigvec(gamma_r[j_ind,:,:], eigrank) for gamma_r in gamma_iter_list])
    return eigvecs

def plot_eigvec_func_em_iters(ax, func, eigrank, dim, gamma_iter_list, gamma_init, j_ind=9, nz=None, color='tab:blue'):
    eigvecs = get_eigvec_em_iters(eigrank, gamma_iter_list, gamma_init, j_ind, nz=nz)
    res = func(eigvecs[:,dim])
    ax.plot(res, color=color, linewidth=2)

def plot_cross_spec_em_iters(ax, i, j, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None):
    cs_real = jnp.array([gamma_init[j_ind_full,i,j].real] + [gamma_r[j_ind,i,j].real for gamma_r in gamma_iter_list])
    cs_imag = jnp.array([gamma_init[j_ind_full,i,j].imag] + [gamma_r[j_ind,i,j].imag for gamma_r in gamma_iter_list])
    ax.plot(cs_real, color='tab:blue', linewidth=2)
    ax.plot(cs_imag, color='tab:red', linewidth=2)

def plot_cross_spec_func_em_iters(ax, func, i, j, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None, color='tab:blue', style='-', width=2):
    cs_real = jnp.array([func(gamma_init[j_ind_full,i,j])] + [func(gamma_r[j_ind,i,j]) for gamma_r in gamma_iter_list])
    ax.plot(cs_real, color=color, linewidth=width, linestyle=style)


