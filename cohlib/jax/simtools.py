import os
import jax.numpy as jnp

from cohlib.utils import gamma_root, pickle_open
from cohlib.jax.gamma_create import k2_full, k2_full_multitarget1, k2_flat, k2_full_testri
from cohlib.jax.dists import naive_estimator
from cohlib.jax.lr_model import LowRankGamma, rotate_eigvecs
from cohlib.jax.plot import get_eigvec, get_eigval

# TODO deprecate all 
def load_gamma(cfg):
    lcfg = cfg.latent
    if cfg.latent.gamma == 'k2-single-10':
        gamma_path = os.path.join(gamma_root(), f"{lcfg.gamma}.pickle")
        gamma_load = pickle_open(gamma_path)
    elif lcfg.gamma == 'k2-flat':
        flow = lcfg.freq_low
        fhigh = lcfg.freq_high
        scale_power = lcfg.scale_power
        k2_flat(flow, fhigh, scale_power)
        gamma_path = os.path.join(gamma_root(), f"k2-flat{flow}-{fhigh}-{scale_power}.pickle")
        gamma_load = pickle_open(gamma_path)
    elif lcfg.gamma == 'k2-full-10':
        flow = lcfg.freq_low
        fhigh = lcfg.freq_high
        sp_target = lcfg.scale_power_target
        sp_offtarget = lcfg.scale_power_offtarget
        k2_full(flow, fhigh, sp_target, sp_offtarget)
        gamma_path = os.path.join(gamma_root(), f"k2-full{flow}-{fhigh}-10-{sp_target}-{sp_offtarget}.pickle")
        gamma_load = pickle_open(gamma_path)
    elif lcfg.gamma == 'k2-full-10-testri':
        flow = lcfg.freq_low
        fhigh = lcfg.freq_high
        sp_target = lcfg.scale_power_target
        sp_offtarget = lcfg.scale_power_offtarget
        k2_full_testri(flow, fhigh, sp_target, sp_offtarget)
        gamma_path = os.path.join(gamma_root(), f"k2-full{flow}-{fhigh}-10-{sp_target}-{sp_offtarget}_testri.pickle")
        gamma_load = pickle_open(gamma_path)
    elif lcfg.gamma == 'k2-full-multitarget1':
        flow = lcfg.freq_low
        fhigh = lcfg.freq_high
        sp_target = lcfg.scale_power_target
        sp_offtarget = lcfg.scale_power_offtarget
        k2_full_multitarget1(flow, fhigh, sp_target, sp_offtarget)
        gamma_path = os.path.join(gamma_root(), f"k2-full{flow}-{fhigh}-mt1-{sp_target}-{sp_offtarget}.pickle")
        gamma_load = pickle_open(gamma_path)
    elif lcfg.gamma == 'k3-temp-to-k2':
        gamma_path = os.path.join(gamma_root(), "k3-temp.pkl")
        gamma_load = pickle_open(gamma_path)
        gamma_load['gamma'] = gamma_load['gamma'][:,:2,:2]
    # elif lcfg.gamma == 'k3-temp':
    #     gamma_path = os.path.join(gamma_root(), "k3-temp.pkl")
    #     gamma_load = pickle_open(gamma_path)
    # elif lcfg.gamma == 'k3-temp-rank1':
    #     gamma_path = os.path.join(gamma_root(), "k3-temp-rank1.pkl")
    #     gamma_load = pickle_open(gamma_path)
    # elif lcfg.gamma == 'k3-temp-rank1-nz9':
    #     gamma_path = os.path.join(gamma_root(), "k3-temp-rank1-nz9.pkl")
    #     gamma_load = pickle_open(gamma_path)
    else:
        gamma_path = os.path.join(gamma_root(), f"{lcfg.gamma}.pkl")
        gamma_load = pickle_open(gamma_path)

    # modify gamma according to config
    gamma_full = gamma_load['gamma']

    gamma_full = gamma_full * lcfg.scale

    gamma_load['gamma'] = gamma_full

    return gamma_load



pp_obs = ['pp_relu', 'pp_log']

def construct_gamma_init(cfg, obs, gamma_load):
    gamma_full = gamma_load['gamma']
    freqs = gamma_load['freqs']
    nz = gamma_load['nonzero_inds']
    nz_target = gamma_load['target_inds']
    K = gamma_full.shape[-1]

    mcfg = cfg.model
    ocfg = cfg.obs

    obs_type = ocfg.obs_type


    if mcfg.init_mod == 1:
        print(f"EM initialization: '{mcfg.init}'")
    else:
        print(f"EM initialization: '{mcfg.init}' - scaled by {mcfg.init_mod}")

    if mcfg.support is not None:
        print(f'Setting model support to {mcfg.support[0]} Hz - {mcfg.support[1]} Hz')
        nz_true = jnp.copy(nz)

        gamma_inv_init_target = jnp.linalg.inv(gamma_full[nz_target,:,:])

        model_nz_filt = (freqs >= mcfg.support[0]) & (freqs <= mcfg.support[1])
        model_freqs = freqs[model_nz_filt]
        model_nonzero_inds = jnp.where(jnp.isin(freqs, model_freqs))[0]

        nz_model = model_nonzero_inds
        Nnz_model = nz_model.size

        gamma_inv_init_flat = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz_model)])*(1/mcfg.scale_init)

        gamma_inv_init = jnp.zeros_like(gamma_full)
        if mcfg.init == 'true-init':
            if nz_model.size == nz_true.size:
                if jnp.all(nz_model == nz_true):
                    # gamma_inv_init_nz = jnp.linalg.inv(gamma_full[nz_true,:,:])
                    gamma_inv_init_nz = jnp.linalg.inv(gamma_full[nz_true,:,:]*mcfg.init_mod)
                    gamma_inv_init = gamma_inv_init.at[nz_true,:,:].set(gamma_inv_init_nz)
                    if mcfg.true_to_flat is not None:
                        print(f'Using true-to-flat: {mcfg.true_to_flat}')
                        if mcfg.true_to_flat == 'off-target':
                            gamma_inv_init = gamma_inv_init.at[nz_target,:,:].set(gamma_inv_init[0,:,:])
                        elif mcfg.true_to_flat == 'target':
                            gamma_init_pre = gamma_full[nz_true,:,:].copy()
                            gamma_init_pre = gamma_init_pre.at[:,0,1].set(0+0j)
                            gamma_init_pre = gamma_init_pre.at[:,1,0].set(0+0j)

                            gamma_target_diag = gamma_full[nz_target,:,:] * jnp.eye(K)
                            gamma_target_diaginv = jnp.linalg.inv(gamma_target_diag*mcfg.init_mod)

                            # gamma_inv_init_nz = jnp.linalg.inv(gamma_init_pre*mcfg.init_mod)

                            # gamma_inv_init = gamma_inv_init.at[nz_true,:,:].set(gamma_inv_init_nz)
                            gamma_inv_init = gamma_inv_init.at[nz_true,:,:].set(gamma_target_diaginv)
                            gamma_init = jnp.linalg.inv(gamma_inv_init[nz_true,:,:])
                            xyz = 543
                        else:
                            raise NotImplementedError
            else:
                raise NotImplementedError
        elif mcfg.init == 'flat-init':
            gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_inv_init_flat)
        elif mcfg.init == 'true-target-flat-offtarget':
            gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_inv_init_flat)
            gamma_inv_init = gamma_inv_init.at[nz_target,:,:].set(gamma_inv_init_target)
        elif mcfg.init == 'empirical-init':
            print('Using empirical (naive) estimate for initialization.')
            if obs_type in pp_obs:
                gamma_empirical = naive_estimator(obs, nz) * 1e6
            elif obs_type == 'gaussian':
                gamma_empirical = naive_estimator(obs, nz) 
            else:
                raise NotImplementedError
            gamma_empirical_inv = jnp.linalg.inv(gamma_empirical)
            # gamma_inv_init = gamma_inv_init.at[nz_model[:-1],:,:].set(gamma_empirical_inv)
            gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_empirical_inv)
        elif mcfg.init == 'empirical-diag-init':
            print('Using empirical (naive) estimate for initialization.')
            if obs_type in pp_obs:
                gamma_empirical = naive_estimator(obs, nz) * 1e6
                print('Setting off-diagonal init terms to zero.')
                gamma_empirical = gamma_empirical * jnp.eye(K)[None,:,:]
                gamma_empirical_inv = jnp.linalg.inv(gamma_empirical)
                gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_empirical_inv)
            else:
                raise NotImplementedError
        elif mcfg.init == 'empirical-flat':
            print('Using empirical (naive) estimate for initialization.')
            if obs_type in pp_obs:
                gamma_empirical = naive_estimator(obs, nz) * 1e6
                print('Setting off-diagonal init terms to zero.')
                gamma_empirical = gamma_empirical * jnp.eye(K)[None,:,:]
                gamma_empirical_inv = jnp.linalg.inv(gamma_empirical)
                gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_empirical_inv)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    # TODO remove 'simple' option here and require 'single' case to be performed through def of support
    else:
        if mcfg.init == 'true-init':
            gamma_inv_init = jnp.linalg.inv(gamma_full[nz,:,:])
        elif mcfg.init == 'flat-init':
            Nnz = nz.size
            gamma_inv_init = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz)])*(1/mcfg.scale_init)
        else:
            raise NotImplementedError
        nz_model = nz
        
    
    gamma_init = jnp.zeros_like(gamma_full)
    gamma_init = gamma_init.at[nz_model,:,:].set(jnp.linalg.inv(gamma_inv_init[nz_model,:,:]))


    return gamma_init, nz_model

# TODO deprecating gamma_load, so this function needs to be reworked as a *method* of LowRankCCN
def construct_gamma_init_rankR(cfg, gamma_load, zs, obs, rank):
    gamma_full = gamma_load['gamma']
    freqs = gamma_load['freqs']
    nz = gamma_load['nonzero_inds']
    nz_target = gamma_load['target_inds']
    K = gamma_full.shape[-1]
    eigvals_true = gamma_load['eigvals']
    eigvecs_true = gamma_load['eigvecs']

    print(f'RANK init: {rank}')

    J = nz_target.size

    mcfg = cfg.model
    ocfg = cfg.obs

    obs_type = ocfg.obs_type

    nz_model = nz
    eigvals_init = jnp.zeros((J,rank))
    eigvecs_init = jnp.zeros((J,K,rank), dtype=complex)

    if cfg.model.init == 'true-init':

        for j in range(J):
            eigvals_init = eigvals_init.at[j,:].set(eigvals_true[j,:])
            eigvecs_init = eigvecs_init.at[j,:,:].set(eigvecs_true[j,:,:])

    
    elif cfg.model.init == 'flat-init':
        # gamma_inv_init_flat = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz_model)])*(1/mcfg.scale_init)

        ones_eigvals = jnp.ones_like(eigvals_init) * mcfg.scale_init * K
        ones_eigvecs = jnp.ones_like(eigvecs_init)/K + 0*1j

        for j in range(J):
            eigvals_init = eigvals_init.at[j,:].set(ones_eigvals[j,:])
            eigvecs_init = eigvecs_init.at[j,:,:].set(ones_eigvecs[j,:,:])
        

    elif mcfg.init == 'oracle-init':
        gamma_oracle = jnp.einsum('jkl,jil->jkil', zs[nz,:,:], zs[nz,:,:].conj()).mean(-1)
        for j in range(J):
            find = nz[j]
            eigvals_oracle_j, eigvecs_oracle_j = jnp.linalg.eigh(gamma_oracle[find,:,:])
            eigvecs_init = eigvecs_init.at[j,:,:].set(eigvecs_oracle_j[:,-rank:][:,::-1])
            eigvals_init = eigvals_init.at[j,:].set(eigvals_oracle_j[-rank:][::-1])


    elif mcfg.init == 'empirical-init':
        print('Using empirical (naive) estimate for initialization.')
        if obs_type in pp_obs:
            gamma_empirical = naive_estimator(obs, nz) * 1e6
        elif obs_type == 'gaussian':
            gamma_empirical = naive_estimator(obs, nz) 
        else:
            raise NotImplementedError
            
        for j, nz_ind in enumerate(nz_model):
            eigvals_empirical_j, eigvecs_empirical_j = jnp.linalg.eigh(gamma_empirical[j,:,:])
            eigvals_init = eigvals_init.at[j,:].set(eigvals_empirical_j[-rank:][::-1])
            eigvecs_init = eigvecs_init.at[j,:,:].set(eigvecs_empirical_j[:,-rank:][:,::-1])




    else:
        raise ValueError

    gamma_init = LowRankGamma(eigvals_init, eigvecs_init, K, freqs, nz)
    return gamma_init, nz_model