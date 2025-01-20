import jax
import jax.numpy as jnp
import jax.random as jr

from cohlib.jax.dists import sample_from_gamma, sample_obs
from cohlib.jax.observations import add0
from cohlib.jax.simtools import load_gamma, construct_gamma_init

from cohlib.jax.models import ToyModel


def gen_data_and_fit_model(cfg):

    lcfg = cfg.latent
    ocfg = cfg.obs
    mcfg = cfg.model

    num_devices = len(jax.devices())
    print(f"NUM_DEVICES={num_devices}")

    gamma_load = load_gamma(cfg)

    gamma_full = gamma_load['gamma']
    freqs = gamma_load['freqs']
    nz_true = gamma_load['nonzero_inds']


    # sample latent and observations according to gamma and observation distribution
    print(f"Sampling {lcfg.L} samples from gamma {lcfg.gamma}; seed = {lcfg.seed}; scale = {lcfg.scale}")
    lrk = jr.key(lcfg.seed)
    zs = sample_from_gamma(lrk, gamma_full, lcfg.L)
    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    obs, obs_params = sample_obs(xs, params)
    obs_type = ocfg.obs_type

    # initialize gamma
    gamma_init, nz_model = construct_gamma_init(cfg, obs, gamma_load)

    # instantiate model and run em
    print(f"Running EM for {mcfg.emiters} iters. Newton iters = {mcfg.maxiter}")

    model = ToyModel()
    model.initialize_latent(gamma_init, freqs, nz_model)
    model.initialize_observations(obs_params, obs_type)
    model.fit_em(obs, mcfg.emiters, mcfg.maxiter)

    gamma_est = model.gamma

    params = {'obs': obs_params,
            'freqs': freqs,
            'true_nonzero_inds': nz_true}

    # save model outputs
    save_dict = {'cfg': cfg, 'gamma': gamma_est, 'params': params, 'gamma_init': gamma_init, 'gamma_true_full': gamma_full, 'track': model.track}

    return save_dict
