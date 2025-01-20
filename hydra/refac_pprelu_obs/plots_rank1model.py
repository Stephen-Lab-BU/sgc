import os
import argparse
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from cohlib.jax.dists import naive_estimator, sample_ccn_rank1
from cohlib.jax.simtools import load_gamma
from cohlib.jax.dists import sample_from_gamma, sample_obs, sample_ccn_rank1
from cohlib.jax.observations import add0
from cohlib.jax.simtools import load_gamma
from cohlib.jax.wrangle import filter_loaded, filter_load_results
from cohlib.jax.plot import (plot_cross_spec_eigval_em_iters, 
                            plot_cross_spec_em_iters, plot_cross_spec_func_em_iters, 
                            plot_eigvals_em_iters, plot_eigvec_func_em_iters,
                            get_eigval, get_eigvec, get_eigvec_em_iters)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('K', nargs='?', type=int, default=3)
    parser.add_argument('init', nargs='?', type=str, default='flat-init')
    parser.add_argument('seed', nargs='?', type=int, default=7)
    parser.add_argument('num_em', nargs='?', type=int, default=25)
    args = parser.parse_args()

    K = args.K
    init_use = args.init
    seed = args.seed
    num_em_iters=args.num_em

    Ls = jnp.array([10, 25, 50])
    mus = jnp.array([10, 25, 50])


    res_paths = [f'/projectnb/stephenlab/jtauber/cohlib/hydra/refac_pprelu_obs/batch/outputs/2024-12-10/seed-{seed}']



    Nnz = 1
    eigrank=1
    jind=9


    print(f'Plotting eigvals for init K: {K}, init: {init_use}, seed: {seed}')
    save_path = '/projectnb/stephenlab/jtauber/cohlib/hydra/refac_pprelu_obs/figs/rank1model'
    if init_use == 'all':
        init_use = 'flat-init'
        res_load = load_res_for_plot(init_use, seed, res_paths, num_em_iters)
        naive_ests, oracle_ests = get_naive_and_oracle(res_load, Ls, mus, K, Nnz)

        name = f'free-k{K}-{init_use}-seed{seed}_var_L_pp_relu_12_10_2024.png'
        savename = os.path.join(save_path, name)
        plot_eigvals_var_L(res_load, naive_ests, oracle_ests, mus, Ls, K, jind, eigrank, Nnz, savename=savename)

        init_use = 'empirical-init'
        res_load = load_res_for_plot(init_use, seed, res_paths, num_em_iters)
        name = f'free-k{K}-{init_use}-seed{seed}_var_L_pp_relu_12_10_2024.png'
        savename = os.path.join(save_path, name)
        plot_eigvals_var_L(res_load, naive_ests, oracle_ests, mus, Ls, K, jind, eigrank, Nnz, savename=savename)

    else:
        res_load = load_res_for_plot(init_use, seed, res_paths, num_em_iters)
        naive_ests, oracle_ests = get_naive_and_oracle(res_load, Ls, mus, K, Nnz)

        name = f'k{K}-{init_use}-seed{seed}_var_L_pp_relu_12_10_2024.png'
        savename = os.path.join(save_path, name)
        plot_eigvals_var_L(res_load, naive_ests, oracle_ests, mus, Ls, K, jind, eigrank, Nnz, savename=savename)


def load_res_for_plot(init_use, seed, res_paths, num_iters):
    lcfg_attrs = {'seed': seed}

    if init_use == 'flat-init':

        init = 'flat-init'
        scale_init = 10000000 
        mcfg_attrs = {'emiters': num_iters,
                    'init': init,
                    'scale_init': scale_init}


    if init_use == 'empirical-init':
        init = 'empirical-init'
        mcfg_attrs = {'emiters': num_iters,
                    'init': init}


    ocfg_attrs = {'obs_type': 'pp_relu', 'seed': seed}

    res_load = filter_load_results(res_paths, lcfg_attrs, mcfg_attrs, ocfg_attrs)

    return res_load


def plot_eigvals_var_L(res_load, naive_ests, oracle_ests, mus, Ls, K, jind, eigrank, Nnz, savename=None):
    gamma_name = f'k{K}-chlg3-relu-rank1-nz9'
    fig, ax = plt.subplots(mus.size, Ls.size, figsize=(12,8), sharex=True, sharey=False)
    jind_full=9
    jind=0

    
    for l, L in enumerate(Ls):
        for a, mu in enumerate(mus):
            lsel = {'L': L, 'gamma': gamma_name}
            msel = {}
            osel = {'alpha': mu}
            res = filter_loaded(res_load, lsel, msel, osel)
            nz_model = res['params']['model_nonzero_inds']
            cfg = res['cfg']

            eig_true = res['eigvals_true'].squeeze()
            eigs_em = jnp.stack([x.eigvals.squeeze() for x in res['track']['gamma_lowrank']])

            gamma_naive = naive_ests[l,a,:,:,:]
            gamma_oracle = oracle_ests[l,a,:,:,:]

            oracle_eigval = get_eigval(gamma_oracle[jind,:,:], eigrank)
            naive_eigval = get_eigval(gamma_naive[jind,:,:], eigrank)

            ax[a,l].plot(eigs_em, color=est_color, linestyle=est_style, linewidth=est_width)
            ax[a,l].axhline(eig_true, linestyle=true_style, color=true_color, linewidth=true_width)
            ax[a,l].axhline(oracle_eigval, linestyle=oracle_style, color=oracle_color, linewidth=oracle_width)
            ax[a,l].axhline(naive_eigval, linestyle=naive_style, color=naive_color, linewidth=naive_width)

            # ax[a,l].axhline(true_eigval, color='k', label='True')
            # ax[a,l].axhline(naive_eigval, linestyle='--', color=naive_color, label='Naive')
            # ax[a,l].axhline(oracle_eigval, linestyle='--', color='tab:green', label='Oracle',linewidth=3)

            if l == 0:
                ax[a,l].set_ylabel(f'alpha={mu}')
            if a == 0:
                ax[a,l].set_title(f'L = {L}')
            if a == mus.size-1:
                ax[a,l].set_xlabel('EM Iter')

    fig.suptitle(f'{nz_model[jind]+1} Hz ' + rf'$\lambda {eigrank}$')
    plt.tight_layout()

    print('saving plot')
    plt.savefig(savename)

naive_color = 'tab:red'
naive_style = 'dashed'
naive_width = 2

true_color = 'k'
true_style = 'solid'
true_width = 2

oracle_color = 'tab:green'
oracle_style = 'dotted'
oracle_width = 2.5

est_color = 'tab:blue'
est_style = 'solid'
est_width = 2


def get_naive_and_oracle(res_load, Ls, mus, K, Nnz):
    naive_ests = jnp.zeros((Ls.size, mus.size, Nnz, K, K), dtype=complex)
    oracle_ests = jnp.zeros((Ls.size, mus.size, Nnz, K, K), dtype=complex)
    gamma_name = f'k{K}-chlg3-relu-rank1-nz9'
    for l, L in enumerate(Ls):
        for a, mu in enumerate(mus):
            print(f'L: {L},  mu: {mu}' )
            lsel = {'L': L,
            'gamma': gamma_name}
            msel = {}
            osel = {'alpha': mu}
            res = filter_loaded(res_load, lsel, msel, osel)

            cfg = res['cfg']
            lcfg = cfg.latent
            ocfg = cfg.obs

            gamma_load = load_gamma(cfg)
            nz_target = gamma_load['target_inds']
            freqs = gamma_load['freqs']
            eigvecs = gamma_load['eigvecs']
            eigvals = gamma_load['eigvals']

            nz_model = nz_target

            lrk = jr.key(lcfg.seed)

            gamma_full_dummytarget = jnp.zeros((freqs.size, K, K), dtype=complex)
            gamma_full_dummytarget = gamma_full_dummytarget.at[nz_target,:,:].set(jnp.eye(K, dtype=complex))

            zs = sample_from_gamma(lrk, gamma_full_dummytarget, lcfg.L)

            for j, ind in enumerate(nz_target):
                zs_target = sample_ccn_rank1(lrk, eigvecs[j,:].squeeze(), eigvals[j].squeeze(), K, lcfg.L)
                zs = zs.at[nz_target,:,:].set(zs_target)

            zs_0dc = jnp.apply_along_axis(add0, 0, zs)
            xs = jnp.fft.irfft(zs_0dc, axis=0)


            obs, obs_params = sample_obs(xs, params)
            naive_est = naive_estimator(obs, nz_model)
            naive_ests = naive_ests.at[l,a,:,:,:].set(naive_est)

            gamma_oracle = jnp.einsum('jkl,jil->jkil', zs[nz_model,:,:], zs[nz_model,:,:].conj()).mean(-1)
            oracle_ests = oracle_ests.at[l,a,:,:,:].set(gamma_oracle)
        
    return naive_ests, oracle_ests

if __name__=="__main__":
    run()