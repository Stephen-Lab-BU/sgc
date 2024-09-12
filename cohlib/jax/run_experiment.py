import os
import multiprocessing

import jax
import jax.numpy as jnp
import jax.random as jr

from cohlib.utils import gamma_root, pickle_open
from cohlib.jax.dists import sample_from_gamma, sample_obs
from cohlib.jax.observations import e_step_par, m_step, add0
from cohlib.jax.gamma_create import k2_full

def gen_data_and_fit_model(cfg):

    lcfg = cfg.latent
    ocfg = cfg.obs
    mcfg = cfg.model

    num_devices = len(jax.devices())
    print(f"NUM_DEVICES={num_devices}")
    print(f"Sampling {lcfg.L} samples from gamma {lcfg.gamma}; seed = {lcfg.seed}; scale = {lcfg.scale}")

    if cfg.latent.gamma == 'k2-single-10':
        gamma_path = os.path.join(gamma_root(), f"{lcfg.gamma}.pickle")
        gamma_load = pickle_open(gamma_path)
    elif lcfg.gamma == 'k2-full-10':
        flow = lcfg.freq_low
        fhigh = lcfg.freq_high
        sp_target = lcfg.scale_power_target
        sp_offtarget = lcfg.scale_power_offtarget
        k2_full(flow, fhigh, sp_target, sp_offtarget)
        gamma_path = os.path.join(gamma_root(), f"k2-full{flow}-{fhigh}-10-{sp_target}-{sp_offtarget}.pickle")
        gamma_load = pickle_open(gamma_path)


    gamma_full = gamma_load['gamma']
    gamma_full = gamma_full * lcfg.scale
    
    freqs = gamma_load['freqs']
    N = freqs.size
    nz = gamma_load['nonzero_inds']
    nz_target = jnp.array([9])
    K = gamma_full.shape[-1]

    gamma_full = gamma_full.at[nz_target,0,1].set(gamma_full[nz_target,0,1]*lcfg.scale_off_diag)
    gamma_full = gamma_full.at[nz_target,1,0].set(gamma_full[nz_target,1,0]*lcfg.scale_off_diag)

    lrk = jr.key(lcfg.seed)
    zs = sample_from_gamma(lrk, gamma_full, lcfg.L)
    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    obs, obs_params = sample_obs(ocfg, xs)
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
        else:
            raise NotImplementedError
        nz = nz_model

    else:
        if mcfg.init == 'true-init':
            gamma_inv_init = jnp.linalg.inv(gamma_full[nz,:,:])
        elif mcfg.init == 'flat-init':
            Nnz = nz.size
            gamma_inv_init = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz)])*(1/mcfg.scale_init)
        else:
            raise NotImplementedError
        
    params = {'obs': obs_params,
              'freqs': freqs,
              'nonzero_inds': nz}
    
    gamma_init = jnp.linalg.inv(gamma_inv_init[nz,:,:])
    track = {'mus': [], 'gamma': []}
    print(f"Running EM for {mcfg.emiters} iters. Newton iters = {mcfg.maxiter}")
    gamma_prev_inv = gamma_inv_init
    for r in range(mcfg.emiters):
        print(f'EM Iter {r+1}')
        if mcfg.track_mus is True:
            mus_all, Upss = e_step_par(obs, gamma_prev_inv, params, obs_type, return_mus=True)
            mus = mus_all[0]
            mus_outer = mus_all[1]
            track['mus'].append(mus)
        else:
            mus_outer, Upss = e_step_par(obs, gamma_prev_inv, params, obs_type)

        gamma_update = m_step(mus_outer, Upss)

        gamma_prev_inv_model = jnp.linalg.inv(gamma_update)
        gamma_prev_inv = jnp.zeros_like(gamma_full)
        gamma_prev_inv = gamma_prev_inv.at[nz,:,:].set(gamma_prev_inv_model)
        track['gamma'].append(gamma_update)

    save_dict = {'gamma': gamma_update, 'params': params, 'gamma_init': gamma_init, 'gamma_true_full': gamma_full, 'track': track}
    return save_dict
