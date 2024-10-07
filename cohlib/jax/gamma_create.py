import os
import jax.numpy as jnp
import jax.random as jr
from cohlib.utils import gamma_root, pickle_save

def k2_single_10():
    N = 500
    K = 2
    nonzero_freqs = jnp.array([10])

    gamma_nonzero = jnp.array([[5,1],[1,5]],dtype=complex) + jnp.array([[0,-1],[1,0]],dtype=complex)*1j
    gamma_full = jnp.zeros((N,K,K), dtype=complex)
    freqs = jnp.arange(1,N+1)
    nonzero_inds = jnp.where(jnp.isin(freqs, nonzero_freqs))[0]

    gamma_full = gamma_full.at[nonzero_inds,:,:].set(gamma_nonzero)

    save_path = os.path.join(gamma_root(), 'k2-single-10.pickle')
    save_dict = {'gamma': gamma_full, 'freqs': freqs, 'nonzero_inds': nonzero_inds}

    pickle_save(save_dict, save_path)

def k2_full(flow, fhigh, scalep_target, scalep_offtarget):
    N = 500
    K = 2
    nonzero_freqs = jnp.arange(flow,fhigh+1)
    Nnz = nonzero_freqs.size

    scale_target = 10**scalep_target
    scale_offtarget = 10**scalep_offtarget

    target_freqs = jnp.array([10])
    target_inds = jnp.where(jnp.isin(nonzero_freqs, target_freqs))[0]

    gamma_target = jnp.array([[5,1],[1,5]],dtype=complex) + jnp.array([[0,-1],[1,0]],dtype=complex)*1j

    gamma_nonzero = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz)])*scale_offtarget

    # diag_mask = jnp.stack([jnp.eye(K) for n in range(Nnz)])
    # rk = jr.key(seed)
    # gamma_nonzero = gamma_nonzero + jr.normal(rk, gamma_nonzero.shape)*diag_mask*0.01*scale
    gamma_nonzero = gamma_nonzero.at[target_inds,:,:].set(gamma_target*scale_target)

    gamma_full = jnp.zeros((N,K,K), dtype=complex)
    freqs = jnp.arange(1,N+1)
    nonzero_inds = jnp.where(jnp.isin(freqs, nonzero_freqs))[0]

    gamma_full = gamma_full.at[nonzero_inds,:,:].set(gamma_nonzero)

    save_path = os.path.join(gamma_root(), f'k2-full{flow}-{fhigh}-10-{scalep_target}-{scalep_offtarget}.pickle')
    save_dict = {'gamma': gamma_full, 'freqs': freqs, 'nonzero_inds': nonzero_inds, 'target_inds': target_inds}

    pickle_save(save_dict, save_path)

def k2_flat(flow, fhigh, scale_power):
    N = 500
    K = 2
    nonzero_freqs = jnp.arange(flow,fhigh+1)
    Nnz = nonzero_freqs.size

    scale_offtarget = 10**scale_power

    gamma_nonzero = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz)])*scale_offtarget

    gamma_full = jnp.zeros((N,K,K), dtype=complex)
    freqs = jnp.arange(1,N+1)
    nonzero_inds = jnp.where(jnp.isin(freqs, nonzero_freqs))[0]

    gamma_full = gamma_full.at[nonzero_inds,:,:].set(gamma_nonzero)

    save_path = os.path.join(gamma_root(), f'k2-flat{flow}-{fhigh}-10-{scale_power}.pickle')
    save_dict = {'gamma': gamma_full, 'freqs': freqs, 'nonzero_inds': nonzero_inds}

    pickle_save(save_dict, save_path)

def k2_full_multitarget1(flow, fhigh, scalep_target, scalep_offtarget):
    N = 500
    K = 2
    nonzero_freqs = jnp.arange(flow,fhigh+1)
    Nnz = nonzero_freqs.size

    scale_target = 10**scalep_target
    scale_offtarget = 10**scalep_offtarget

    target_freqs = jnp.array([1, 13, 28])
    mt_scales = [0.5, 1, 0.8]
    target_inds = jnp.where(jnp.isin(nonzero_freqs, target_freqs))[0]

    gamma_target = jnp.array([[5,1],[1,5]],dtype=complex) + jnp.array([[0,-1],[1,0]],dtype=complex)*1j

    gamma_nonzero = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz)])*scale_offtarget

    # diag_mask = jnp.stack([jnp.eye(K) for n in range(Nnz)])
    # rk = jr.key(seed)
    # gamma_nonzero = gamma_nonzero + jr.normal(rk, gamma_nonzero.shape)*diag_mask*0.01*scale
    for i, j in enumerate(target_freqs):
        mt_scale = mt_scales[i]
        gamma_nonzero = gamma_nonzero.at[j,:,:].set(gamma_target*scale_target*mt_scale)

    gamma_full = jnp.zeros((N,K,K), dtype=complex)
    freqs = jnp.arange(1,N+1)
    nonzero_inds = jnp.where(jnp.isin(freqs, nonzero_freqs))[0]

    gamma_full = gamma_full.at[nonzero_inds,:,:].set(gamma_nonzero)

    save_path = os.path.join(gamma_root(), f'k2-full{flow}-{fhigh}-mt1-{scalep_target}-{scalep_offtarget}.pickle')
    save_dict = {'gamma': gamma_full, 'freqs': freqs, 'nonzero_inds': nonzero_inds, 'target_inds': target_inds}

    pickle_save(save_dict, save_path)