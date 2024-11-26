import jax
import jax.numpy as jnp
import jax.random as jr

from cohlib.jax.dists import sample_from_gamma, sample_obs, sample_ccn_rank1
from cohlib.jax.observations import add0
from cohlib.jax.simtools import load_gamma, construct_gamma_init

from cohlib.jax.models import ToyModel


def gen_data_and_fit_model_rank1(cfg):

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
    eigvec = gamma_load['eigvec']
    eigval = gamma_load['eigval']


    # sample latent and observations according to gamma and observation distribution
    print(f"Sampling {lcfg.L} samples from gamma {lcfg.gamma}; seed = {lcfg.seed}; scale = {lcfg.scale}")
    if ocfg.obs_type == 'pp_relu' or ocfg.obs_type == 'pp_log':
        print(f'alpha = {ocfg.alpha}')
    if ocfg.obs_type == 'gaussian':
        print(f'obs var = {ocfg.ov1}e{ocfg.ov2}')
    lrk = jr.key(lcfg.seed)

    zs_target = sample_ccn_rank1(lrk, eigvec, eigval, K, lcfg.L)
    gamma_full_dummytarget = gamma_full.copy()
    gamma_full_dummytarget = gamma_full_dummytarget.at[nz_target,:,:].set(jnp.eye(K, dtype=complex))

    zs = sample_from_gamma(lrk, gamma_full_dummytarget, lcfg.L)
    zs = zs.at[nz_target,:,:].set(zs_target)

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    obs, obs_params = sample_obs(ocfg, xs)
    obs_type = ocfg.obs_type

    # initialize gamma
    gamma_init, nz_model = construct_gamma_init(cfg, obs, gamma_load)

    # instantiate model and run em
    print(f"Running EM for {mcfg.emiters} iters. Newton iters = {mcfg.maxiter}")

    # TODO: handle m_step options in better way
    if mcfg.m_step_option == 'standard':
        m_step_params = None
    elif mcfg.m_step_option == 'low-rank':
        m_step_params = {'rank': mcfg.m_step_rank}

    model = ToyModel()
    model.initialize_latent(gamma_init, freqs, nz_model)
    model.initialize_observations(obs_params, obs_type)
    model.fit_em(obs, mcfg.emiters, mcfg.maxiter, m_step_option=mcfg.m_step_option, m_step_params=m_step_params)

    gamma_est = model.gamma

    params = {'obs': obs_params,
            'freqs': freqs,
            'true_nonzero_inds': nz_true,
            'model_nonzero_inds': nz_model}

    # save model outputs
    save_dict = {'cfg': cfg, 'gamma': gamma_est, 'params': params, 'gamma_init': gamma_init, 'gamma_true_full': gamma_full, 'track': model.track}

    return save_dict