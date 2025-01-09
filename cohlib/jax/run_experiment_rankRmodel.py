import jax
import jax.numpy as jnp
import jax.random as jr

from cohlib.jax.dists import sample_from_gamma, sample_obs, sample_ccn_rank1
from cohlib.jax.observations import add0
from cohlib.jax.simtools import load_gamma, construct_gamma_init_rankR

from cohlib.jax.lr_model import LowRankToyModel, rotate_eigvecs


# TODO separate into two separate functions - gen_data and fit_model
def gen_data_and_fit_model_rankRm(cfg):

    lcfg = cfg.latent
    ocfg = cfg.obs
    mcfg = cfg.model

    num_devices = len(jax.devices())
    print(f"NUM_DEVICES={num_devices}")

    # TODO simplifiy design (read below)
    # let's just have gamma constructed on the fly instead of storing it... that way we don't have to keep track of all the names etc
    # gamma will always be saved with the experiment, and will be deterministic based on relevant parameters
    # ah - maybe that's the issue... 
    # instead, let's make use of the low-rank CCN class we made and just require in our experiment script that gamma and metadata are all valid
    # so instead of load_gamma we can just have an abstract procedure that produces a gamma, and then we can drop in any method to do so
    # importantly, we want to be able to declare how this happens using hydra cfg with minimal conceptual overhead


    gamma_load = load_gamma(cfg)

    gamma_full = gamma_load['gamma']
    K = gamma_full.shape[-1]
    freqs = gamma_load['freqs']
    nz_true = gamma_load['nonzero_inds']
    nz_target = gamma_load['target_inds']
    J = nz_true.size

    # TODO these should just be an intrinsic part of requirement outlined above
    # change all gamma generating functions to create CCN class instead of properties (or wrap that up somewhere that is not here)
    eigvecs_true = gamma_load['eigvecs']
    eigvals_true = gamma_load['eigvals']
    true_rank = eigvals_true.shape[1]


    # sample latent and observations according to gamma and observation distribution
    # TODO wrap this into a sampling method for distribution object 
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

    obs, obs_params = sample_obs(ocfg, xs)
    obs_type = ocfg.obs_type

    # TODO make gamma initilization part of model object 
    # NOTE in general need to strongly separate synthetic dist information and model information  

    # initialize gamma
    # gamma_init, nz_model = construct_gamma_init_lowrank(cfg, obs, gamma_load)

    # instantiate model and run em
    print("Fitting Low-Rank Model: rank = 1")
    print(f"Running EM for {mcfg.emiters} iters. Newton iters = {mcfg.maxiter}")

    # TODO for model, have set of parameters, and set of bools that determines whether parameters to be fit or not
    # NOTE basically - better UI for interacting with model 
    fixed_u_mods = ['fixed_u_true', 'fixed_u_oracle']
    fixed_eigval_mods = ['fixed_eigval_true', 'fixed_eigval_oracle']

    # TODO make standalone debug section of config and *model* - some way to isolate these kind of tweaks from rest of code
    # would be handy
    ts_flag = mcfg.ts_flag
    ts_flag2 = mcfg.ts_flag2
    print(f'Using mods {ts_flag} and {ts_flag2}')
    if ts_flag2 != 'eigh_est':
        print('R>1 only implemented for eigh est m-step.')
        assert NotImplementedError
    m_step_seed = mcfg.m_step_seed
    m_step_init = mcfg.m_step_init

    # TODO test rank > 1 behavior after above refactors and add functionality where needed
    model_rank = 1

    # TODO refactor `construct` function
    gamma_init, nz_model = construct_gamma_init_rankR(cfg, gamma_load, zs, obs, model_rank)

    # TODO absolute mess - handle this better (at least wrap it)
    # NOTE handling params as fixed/not by default would take care of this
    if ts_flag == 'fixed_u_true':
        eigvals_fixed = None
        eigvecs_fixed = jnp.zeros((J,K,model_rank),dtype=complex)
        eigvecs_fixed = eigvecs_fixed.at[:,:,:true_rank].set(eigvecs_true)
    elif ts_flag == 'fixed_u_oracle':
        eigvals_fixed = None
        eigvecs_oracle = jnp.zeros((J,K,model_rank),dtype=complex)
        gamma_oracle = jnp.einsum('jkl,jil->jkil', zs[nz_model,:,:], zs[nz_model,:,:].conj()).mean(-1)
        for j in range(J):
            find = nz_model[j]
            _, eigvecs_oracle_j_fr = jnp.linalg.eigh(gamma_oracle[find,:,:].squeeze())
            eigvecs_oracle = eigvecs_oracle.at[j,:,:].set(eigvecs_oracle_j_fr[:,-model_rank:][:,::-1])
        eigvecs_fixed = eigvecs_oracle
    elif ts_flag == 'fixed_eigval_true':
        eigvecs_fixed = None
        eigvals_fixed = jnp.zeros((J,model_rank),dtype=complex)
        eigvals_fixed = eigvals_fixed.at[:,:true_rank].set(eigvals_true)
    elif ts_flag == 'fixed_eigval_oracle':
        eigvecs_fixed = None
        eigvals_oracle = jnp.zeros((J,model_rank),dtype=complex)
        gamma_oracle = jnp.einsum('jkl,jil->jkil', zs[nz_model,:,:], zs[nz_model,:,:].conj()).mean(-1)
        for j in range(J):
            find = nz_model[j]
            eigvals_oracle_j_fr, _ = jnp.linalg.eigh(gamma_oracle[find,:,:].squeeze())
            eigvals_oracle = eigvals_oracle.at[j,:].set(eigvals_oracle_j_fr[-model_rank:][::-1])
        eigvals_fixed = eigvals_oracle 
    else:
        eigvecs_fixed = None
        eigvals_fixed = None

    # TODO make subset of general parameters to pass when using fit_em
    m_step_params = {'m_step_seed': m_step_seed,
                     'init_type': m_step_init, 
                     'ts_flag': ts_flag,
                     'ts_flag2': ts_flag2,
                     'u': eigvecs_fixed,
                     'eigvals': eigvals_fixed}

    print('eigvecs_fixed:')
    print(eigvecs_fixed)


    if ts_flag in fixed_u_mods:
        gamma_init.eigvecs = eigvecs_fixed
    if ts_flag in fixed_eigval_mods:
        gamma_init.eigvals = eigvals_fixed

    print('gamma_init eigvecs:')
    gamma_init.eigvecs = rotate_eigvecs(gamma_init.eigvecs)
    print(gamma_init.eigvecs)
    print('gamma_init eigvals:')
    print(gamma_init.eigvals)

    # NOTE this right here looks great - simplify m_step kw's though, and ideally just have dict of params or params class
    model = LowRankToyModel()
    model.initialize_latent(gamma_init, freqs, nz_model)
    model.initialize_observations(obs_params, obs_type)
    model.fit_em(obs, mcfg.emiters, mcfg.maxiter, m_step_option=mcfg.m_step_option, m_step_params=m_step_params)

    gamma_est = model.gamma_lowrank

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