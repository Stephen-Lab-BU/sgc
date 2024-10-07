import os
import multiprocessing

import jax
import jax.numpy as jnp
import jax.random as jr

from cohlib.utils import gamma_root, pickle_open
from cohlib.jax.dists import sample_from_gamma, sample_obs, naive_estimator
from cohlib.alg.em_gaussian_obs import fit_gaussian_model
from cohlib.jax.observations import e_step_par, m_step, add0
from cohlib.jax.gamma_create import k2_full, k2_full_multitarget1
from cohlib.jax.ts_gaussian import JvOExp
from cohlib.alg.em_sgc import construct_Gamma_full_real, deconstruct_Gamma_full_real

def gen_data_and_fit_model_ts_scipy(cfg, inverse_correction):

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
    elif lcfg.gamma == 'k2-full-multitarget1':
        flow = lcfg.freq_low
        fhigh = lcfg.freq_high
        sp_target = lcfg.scale_power_target
        sp_offtarget = lcfg.scale_power_offtarget
        k2_full_multitarget1(flow, fhigh, sp_target, sp_offtarget)
        gamma_path = os.path.join(gamma_root(), f"k2-full{flow}-{fhigh}-mt1-{sp_target}-{sp_offtarget}.pickle")
        gamma_load = pickle_open(gamma_path)
    else:
        raise NotImplementedError


    gamma_full = gamma_load['gamma']
    gamma_full = gamma_full * lcfg.scale
    
    freqs = gamma_load['freqs']
    N = freqs.size
    nz = gamma_load['nonzero_inds']
    nz_target = gamma_load['target_inds']
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
        elif mcfg.init == 'empirical-init':
            print('Using empirical (naive) estimate for initialization.')
            if obs_type != 'gaussian':
                gamma_empirical = naive_estimator(obs, nz) * 1e6
                gamma_empirical_inv = jnp.linalg.inv(gamma_empirical)
                gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_empirical_inv)
            else:
                raise NotImplementedError
        elif mcfg.init == 'empirical-diag-init':
            print('Using empirical (naive) estimate for initialization.')
            if obs_type != 'gaussian':
                gamma_empirical = naive_estimator(obs, nz) * 1e6
                print('Setting off-diagonal init terms to zero.')
                gamma_empirical = gamma_empirical * jnp.eye(K)[None,:,:]
                gamma_empirical_inv = jnp.linalg.inv(gamma_empirical)
                gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_empirical_inv)
            else:
                raise NotImplementedError
        elif mcfg.init == 'empirical-flat':
            print('Using empirical (naive) estimate for initialization.')
            if obs_type != 'gaussian':
                gamma_empirical = naive_estimator(obs, nz) * 1e6
                print('Setting off-diagonal init terms to zero.')
                gamma_empirical = gamma_empirical * jnp.eye(K)[None,:,:]
                gamma_empirical_inv = jnp.linalg.inv(gamma_empirical)
                gamma_inv_init = gamma_inv_init.at[nz_model,:,:].set(gamma_empirical_inv)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        nz = nz_model

    # TODO remove 'simple' option here and require 'single' case to be performed through def of support
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
    print(f"Running EM for {mcfg.emiters} iters. Newton iters = {mcfg.maxiter}")

    old_model_load = load_old()
    Wv = old_model_load['Wv']
    obs_var = obs_params['obs_var']


    sample_length = Wv.shape[0]
    num_J_vars = Wv.shape[1]
    gamma_inv_oldformat = 4*construct_Gamma_full_real(gamma_inv_init[nz,:,:], 
                        K, num_J_vars, invert=False)
    inits = {
        'Gamma_inv_init': gamma_inv_oldformat,
        # 'Gamma_inv_init': true_init,
        # 'Gamma_inv_init': sampletrue_init,
        'mu': 0,
        'Gamma_true': gamma_full
        }

    tapers = None
    invQ = jnp.diag(jnp.ones(sample_length)*(1/obs_var))
    invQs = [invQ for k in range(K)]

    print(f'Correction to inverse conversion error: {inverse_correction}')
    obs_oldorder = obs.swapaxes(0,-1)
    obs_grouped = [obs_oldorder[:,None,k,:] for k in range(K)]
    gamma_est, gamma_est_tapers, track = fit_gaussian_model(obs_grouped, Wv, inits, tapers, invQs, etype='approx', num_em_iters=mcfg.emiters, 
                max_approx_iters=0, track=True, inverse_correction=inverse_correction)

    method = 'scipy'
    if inverse_correction is True:
        method = 'scipy-oldmod'
    else:
        method = 'scipy-old'



    # save flag so that can differentiate ts vs non ts runs
    save_dict = {'ts_run': True, 'method': method, 'gamma': gamma_est, 'params': params, 'gamma_init': gamma_init, 'gamma_true_full': gamma_full, 'track': track}

    # end
    return save_dict

def load_old(mu=0.0, K=2, L=25, sample_length=1000, C=1, ov1=1.0, seed=8, etype="approx", hess_mod=False):
    exp_path = '/projectnb/stephenlab/jtauber/cohlib/experiments/gaussian_observations'
    ov2 = float(-1)
    if hess_mod is True:
        model_path = f'{exp_path}/saved/fitted_models/scale_hess_mod_jax_comp_simple_synthetic_gaussian_em20_{K}_{L}_{sample_length}_{C}_{mu}_{ov1}_{ov2}_{seed}_fitted_{etype}.pkl'
        model_load = pickle_open(model_path)
    else:
        model_path = f'{exp_path}/saved/fitted_models/scale_mod_jax_comp_simple_synthetic_gaussian_em20_{K}_{L}_{sample_length}_{C}_{mu}_{ov1}_{ov2}_{seed}_fitted_{etype}.pkl'
        model_load = pickle_open(model_path)

    return model_load
