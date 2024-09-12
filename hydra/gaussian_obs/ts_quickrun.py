import os
import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
from cohlib.jax.dists import sample_from_gamma
from cohlib.jax.gaussian_obs import add0

from cohlib.utils import gamma_root, pickle_open, pickle_save
from cohlib.jax.ts_gaussian import load_results, JvOExp


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('obs_var2', nargs='?', type=float, default=3)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    flow = 1
    fhigh = 50
    scalep_target = 5
    scalep_offtarget = 1

    ov1 = 1
    ov2 = float(args.obs_var2)
    obs_var = ov1 * 10**ov2

    gammas_path = gamma_root()
    gamma_load = pickle_open(f'{gammas_path}/k2-full{flow}-{fhigh}-10-{scalep_target}-{scalep_offtarget}.pickle')
    gamma_full = gamma_load['gamma']
    freqs = gamma_load['freqs']
    nz = gamma_load['nonzero_inds']
    K = gamma_full.shape[1]

    scale_init = 10
    res_paths = ['/projectnb/stephenlab/jtauber/cohlib/hydra/batch_submit/outputs/2024-08-07']

    L = 25
    init = 'flat-init'
    emiters = 20
    scale_init = 10
    ovs_sel = None

    supp = [0, 50]
    res_load = load_results(res_paths, ovs_sel, L=L, init=init, emiters=emiters, supp=supp, scale_init=scale_init)
    cfg = res_load[ov2]['cfg']
    lcfg = cfg.latent
    ocfg = cfg.obs

    lrk = jr.key(lcfg.seed)
    ork = jr.key(ocfg.seed)

    Nnz = nz.size

    gamma_inv_true = jnp.zeros_like((gamma_full))
    gamma_inv_true = gamma_inv_true.at[nz,:,:].set(jnp.linalg.inv(gamma_full[nz,:,:]))

    nz_power_init = 10
    gamma_inv_flat = jnp.zeros_like((gamma_full))
    gamma_inv_flat_nz = jnp.stack([jnp.eye(K, dtype=complex) for _ in range(Nnz)])*(1/nz_power_init)
    gamma_inv_flat = gamma_inv_flat.at[nz,:,:].set(gamma_inv_flat_nz)

    L = 25

    zs = sample_from_gamma(lrk, gamma_full, L)
    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)
    obs = xs + jr.normal(ork, xs.shape)*jnp.sqrt(obs_var)

    old_model_load = load_old(ov2)
    Wv = old_model_load['Wv']


    params_jax = {'freqs': freqs, 'nonzero_inds': nz}
    params_old = {'Wv': Wv, 'nonzero_inds': nz}

    gamma_inv_use = gamma_inv_flat
    print('Running jax v old first E-step with flat init.')
    jax_exp = JvOExp(obs, gamma_inv_use, obs_var, params_jax, 'jax', track=True)
    old_exp = JvOExp(obs, gamma_inv_use, obs_var, params_old, 'oldmod', track=True)
    # old_exp = JvOExp(obs, gamma_inv_use, obs_var, params_old, 'old', track=True)

    jax_exp.e_step(10)
    old_exp.e_step(10)


    res = {'zs_true': zs, 'obs': obs, 'jax': jax_exp, 'old': old_exp}
    pickle_save(res, f'ts_output/{ov2}_{lcfg.seed}_flatinit_mus_oldmod.pickle')

def load_old(ov2, mu=0.0, K=2, L=25, sample_length=1000, C=1, ov1=1.0, seed=8, etype="approx", hess_mod=False):
    exp_path = '/projectnb/stephenlab/jtauber/cohlib/experiments/gaussian_observations'
    ov2 = float(ov2)
    if hess_mod is True:
        model_path = f'{exp_path}/saved/fitted_models/scale_hess_mod_jax_comp_simple_synthetic_gaussian_em20_{K}_{L}_{sample_length}_{C}_{mu}_{ov1}_{ov2}_{seed}_fitted_{etype}.pkl'
        model_load = pickle_open(model_path)
    else:
        model_path = f'{exp_path}/saved/fitted_models/scale_mod_jax_comp_simple_synthetic_gaussian_em20_{K}_{L}_{sample_length}_{C}_{mu}_{ov1}_{ov2}_{seed}_fitted_{etype}.pkl'
        model_load = pickle_open(model_path)

    return model_load

if __name__ == '__main__':
    run()


