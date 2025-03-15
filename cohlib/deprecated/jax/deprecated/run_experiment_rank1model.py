import jax
import jax.numpy as jnp
import jax.random as jr

from cohlib.jax.dists import sample_from_gamma, sample_obs, sample_ccn_rank1
from cohlib.jax.observations import add0
from cohlib.jax.simtools import load_gamma, construct_gamma_init_rank1

from cohlib.jax.lr_model import LowRankToyModel
from cohlib.jax.plot import get_eigvec


def gen_data_and_fit_model_rank1m(cfg):

    lcfg = cfg.latent
    ocfg = cfg.obs
    mcfg = cfg.model

    num_devices = len(jax.devices())
    print(f"NUM_DEVICES={num_devices}")

    gamma_load = load_gamma(cfg)

    gamma_full = gamma_load['gamma']
    K = gamma_full.shape[-1]
    freqs = gamma_load['freqs']
    nz_true = gamma_load['nonzero_inds']
    nz_target = gamma_load['target_inds']
    eigvecs_true = gamma_load['eigvecs']
    eigvals_true = gamma_load['eigvals']


    # sample latent and observations according to gamma and observation distribution
    print(f"Sampling {lcfg.L} samples from gamma {lcfg.gamma}; seed = {lcfg.seed}; scale = {lcfg.scale}")
    if ocfg.obs_type == 'pp_relu' or ocfg.obs_type == 'pp_log':
        print(f'alpha = {ocfg.alpha}')
    if ocfg.obs_type == 'gaussian':
        print(f'obs var = {ocfg.ov1}e{ocfg.ov2}')
    lrk = jr.key(lcfg.seed)

    gamma_full_dummytarget = gamma_full.copy()
    gamma_full_dummytarget = gamma_full_dummytarget.at[nz_target,:,:].set(jnp.eye(K, dtype=complex))

    zs = sample_from_gamma(lrk, gamma_full_dummytarget, lcfg.L)

    for j, ind in enumerate(nz_target):
        zs_target = sample_ccn_rank1(lrk, eigvecs_true[j,:].squeeze(), eigvals_true[j].squeeze(), K, lcfg.L)
        zs = zs.at[nz_target,:,:].set(zs_target)

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    obs, obs_params = sample_obs(xs, params)
    obs_type = ocfg.obs_type

    # initialize gamma
    # gamma_init, nz_model = construct_gamma_init_lowrank(cfg, obs, gamma_load)

    # instantiate model and run em
    print("Fitting Low-Rank Model: rank = 1")
    print(f"Running EM for {mcfg.emiters} iters. Newton iters = {mcfg.maxiter}")

    fixed_u_mods = ['fixed_u_true', 'fixed_u_oracle']
    fixed_eigval_mods = ['fixed_eigval_true', 'fixed_eigval_oracle']
    ts_flag = mcfg.ts_flag
    ts_flag2 = mcfg.ts_flag2
    print(f'Using mods {ts_flag} and {ts_flag2}')
    m_step_seed = mcfg.m_step_seed
    m_step_init = mcfg.m_step_init


    if ts_flag == 'fixed_u_true':
        eigvecs_fixed = eigvecs_true
        eigvals_fixed = None
    elif ts_flag == 'fixed_u_oracle':
        gamma_oracle = jnp.einsum('jkl,jil->jkil', zs[nz_target,:,:], zs[nz_target,:,:].conj()).mean(-1)
        eigvec_oracle = get_eigvec(gamma_oracle[nz_target,:,:].squeeze(), 1).squeeze()
        eigvecs_oracle = jnp.zeros_like(eigvecs_true)
        eigvecs_oracle = eigvecs_oracle.at[0,:,0].set(eigvec_oracle)
        eigvecs_fixed = eigvecs_oracle 
        eigvals_fixed = None
    elif ts_flag == 'fixed_eigval_true':
        eigvals_fixed = eigvals_true
        eigvecs_fixed = None
    elif ts_flag == 'fixed_eigval_oracle':
        eigvals_oracle = jnp.zeros_like(eigvals_true)
        eigval_oracle = get_eigvec(gamma_oracle[nz_target,:,:].squeeze(), 1).squeeze()
        eigvals_oracle = eigvals_oracle.at[0,0].set(eigval_oracle)
        eigvals_fixed = eigvals_oracle 
    else:
        eigvecs_fixed = None
        eigvals_fixed = None

    m_step_params = {'m_step_seed': m_step_seed,
                     'init_type': m_step_init, 
                     'ts_flag': ts_flag,
                     'ts_flag2': ts_flag2,
                     'u': eigvecs_fixed,
                     'eigvals': eigvals_fixed}

    print('eigvecs_fixed:')
    print(eigvecs_fixed)
    gamma_init, nz_model = construct_gamma_init_rank1(cfg, gamma_load, zs, obs)


    if ts_flag in fixed_u_mods:
        gamma_init.eigvecs = eigvecs_fixed
    if ts_flag in fixed_eigval_mods:
        gamma_init.eigvals = eigvals_fixed

    print('gamma_init eigvecs:')
    print(gamma_init.eigvecs)

    model = LowRankToyModel()
    model.initialize_latent(gamma_init, freqs, nz_model)
    model.initialize_observations(obs_params, obs_type)
    model.fit_em(obs, mcfg.emiters, mcfg.maxiter, m_step_option=mcfg.m_step_option, m_step_params=m_step_params)

    gamma_est = model.lrccn

    params_save = {'obs': obs_params,
            'freqs': freqs,
            'true_nonzero_inds': nz_true,
            'model_nonzero_inds': nz_model,
            'm_step_seed': m_step_seed,
            'ts_flag': ts_flag,
            'ts_flag2': ts_flag2,
            'eigvals_fixed': eigvals_fixed,
            'eigvecs_fixed': eigvecs_fixed}

    print('final eigvecs_fixed:')
    print(eigvecs_fixed)

    # save model outputs
    save_dict = {'cfg': cfg, 'gamma_lowrank': gamma_est, 'params': params_save, 'gamma_init': gamma_init, 'eigvals_true': eigvals_true, 'eigvecs_true': eigvecs_true, 'track': model.track}

    return save_dict