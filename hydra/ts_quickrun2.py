import os
import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
from cohlib.jax.dists import sample_from_gamma
from cohlib.jax.gaussian_obs import add0

from cohlib.utils import gamma_root, pickle_open, pickle_save
from cohlib.jax.ts import load_results, JvOExp


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('obs_var2', nargs='?', type=float, default=3)
    parser.add_argument('trial', nargs='?', type=int, default=0)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    ov2 = args.obs_var2
    seed = 7
    # res = pickle_open(f'ts_output/{ov2}_{seed}_trueinit_mus.pickle')
    res = pickle_open(f'ts_output/{ov2}_{seed}_flatinit_mus.pickle')
    # res = pickle_open(f'ts_output/{ov2}_{seed}_flatinit_mus_oldmod.pickle')

    jax_exp = res['jax']
    old_exp = res['old']
    num_iters = len(old_exp.track_data[0].track_zs) 
    nz = jnp.arange(50)

    trial = args.trial
    # zs_use = res['zs_true'][nz,:,trial]
    # # old_r = old_exp.track_data[trial].track_grad[r]
    # jax_r, _, _ = jax_exp.eval_cost(trial, zs_use)
    # old_r, _, _ = old_exp.eval_cost(trial, zs_use)
    # print(f' jax cost: {jax_r.real}, old cost: {old_r.real} jax/old: {jax_r.real / old_r.real}')

    # for r in range(num_iters):
    old_mus = old_exp.mus[:,:,trial]
    jax_mus = jax_exp.mus[:,:,trial]

    zs_use = old_mus

    ocost, ograd, ohess = old_exp.eval_cost(trial, zs_use)
    jcost, jgrad, jhess = jax_exp.eval_cost(trial, zs_use)
    # j
    # for r in [4]:
    #     zs_use = old_exp.track_data[trial].track_zs[r]
    #     # old_r = old_exp.track_data[trial].track_grad[r]
    #     jax_r, _, _ = jax_exp.eval_cost(trial, zs_use)
    #     old_r, _, _ = old_exp.eval_cost(trial, zs_use)
    #     print(f' jax cost: {jax_r.real}, old cost: {old_r.real} jax/old: {jax_r.real / old_r.real}')


if __name__ == '__main__':
    run()


