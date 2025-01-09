import os
import argparse
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
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
    parser.add_argument('eigrank', nargs='?', type=int, default=1)
    parser.add_argument('num_em', nargs='?', type=int, default=25)
    args = parser.parse_args()

    K = args.K
    init_use = args.init
    seed = args.seed

    # Ls = jnp.array([10, 25, 50])
    # ov_pairs = jnp.array([(1, -1), (5.0, -1), 
    #                     (1, 0), (5.0, 0)])

    Ls = jnp.array([10, 25])
    ov_pairs = jnp.array([(1, 0), (5.0, 0)])

    gamma_name = f'k{K}-chlg4-rotate-gaussian-rank1-nz9'

    # res_paths = [f'/projectnb/stephenlab/jtauber/cohlib/hydra/gaussian_obs_postfix/outputs/2024-12-19-fixed_eigvec_true-eigh_est/{gamma_name}/seed-{seed}']
    # save_dir_gamma = f'/projectnb/stephenlab/jtauber/cohlib/hydra/gaussian_obs_postfix/figs/local_r1new/fixed-eigvec-true/eigh_est/{gamma_name}/'

    # res_paths = [f'/projectnb/stephenlab/jtauber/cohlib/hydra/gaussian_obs_postfix/outputs/2024-12-19-fixed_eigval_true-eigh_est/{gamma_name}/seed-{seed}']
    # save_dir_gamma = f'projectnb/stephenlab/jtauber/cohlib/hydra/gaussian_obs_postfix/figs/local_r1new/fixed-eigval-true/eigh_est/{gamma_name}/'

    date = '2024-12-20'
    ts_flag = 'fixed_eigval_true'
    ts_flag2 = 'eigh_est'
    addendum = 'rotatem'
    m_step_init = 'False'
    res_paths = [f'/projectnb/stephenlab/jtauber/cohlib/hydra/gaussian_obs_postfix/outputs/{date}-rankR/{ts_flag}/{ts_flag2}/{gamma_name}/seed-{seed}/{m_step_init}']
    save_dir_gamma = f'/projectnb/stephenlab/jtauber/cohlib/hydra/gaussian_obs_postfix/figs/local_r2/{ts_flag}-{addendum}/{ts_flag2}/{gamma_name}/{m_step_init}'

    Nnz = 1
    eigrank = args.eigrank
    dims = [0,1,2]
    jind=9

    current_date = pd.Timestamp.today().date()

    print(f'Plotting eigvals for init K: {K}, init: {init_use}, seed: {seed}')


    if init_use == 'all':
        inits = ['flat-init', 'empirical-init']
    else:
        inits = [init_use]

    res_load = load_res_for_plot(inits[0], seed, res_paths)
    print(len(res_load))
    naive_ests, oracle_ests = get_naive_and_oracle(res_load, Ls, ov_pairs, K, Nnz, gamma_name)

    for init in inits:
        res_load = load_res_for_plot(init, seed, res_paths)
        name = f'k{K}-{init}-seed{seed}_var_L_gaussian_{current_date}.png'

        
        save_dir_init = os.path.join(save_dir_gamma, init)
        if not os.path.exists(save_dir_init):
            os.makedirs(save_dir_init)

        savename = os.path.join(save_dir_init, f'eigval{eigrank}-'+name)
        plot_eigvals_var_L(res_load, naive_ests, oracle_ests, ov_pairs, Ls, K, jind, eigrank, Nnz, gamma_name=gamma_name, savename=savename)

        for dim in dims:
            save_dir_eigrank = os.path.join(save_dir_init, f'eig{eigrank}')
            if not os.path.exists(save_dir_eigrank):
                os.makedirs(save_dir_eigrank)
            savename = os.path.join(save_dir_eigrank, f'eigvec-phase-dim{dim}-'+name)
            plot_func_eigvec_var_L(jnp.angle, 'Phase', dim, res_load, naive_ests, oracle_ests, ov_pairs, Ls, K, jind, eigrank, Nnz, gamma_name=gamma_name, savename=savename)
            
            savename = os.path.join(save_dir_eigrank, f'eigvec-mag-dim{dim}-'+name)
            plot_func_eigvec_var_L(jnp.abs, 'Mag', dim, res_load, naive_ests, oracle_ests, ov_pairs, Ls, K, jind, eigrank, Nnz, gamma_name=gamma_name, savename=savename)

            savename = os.path.join(save_dir_eigrank, f'eigvec-real-dim{dim}-'+name)
            plot_func_eigvec_var_L(jnp.real, 'Real', dim, res_load, naive_ests, oracle_ests, ov_pairs, Ls, K, jind, eigrank, Nnz, gamma_name=gamma_name, savename=savename)
            
            savename = os.path.join(save_dir_eigrank, f'eigvec-imag-dim{dim}-'+name)
            plot_func_eigvec_var_L(jnp.imag, 'Imag', dim, res_load, naive_ests, oracle_ests, ov_pairs, Ls, K, jind, eigrank, Nnz, gamma_name=gamma_name, savename=savename)



def load_res_for_plot(init_use, seed, res_paths, num_iters=25):
    lcfg_attrs = {'seed': seed}

    if init_use == 'flat-init':

        init = 'flat-init'
        scale_init = 100
        mcfg_attrs = {'emiters': num_iters,
                    'init': init,
                    'scale_init': scale_init}
    else:
        mcfg_attrs = {'emiters': num_iters,
                    'init': init_use}


    ocfg_attrs = {'obs_type': 'gaussian', 'seed': seed}

    res_load = filter_load_results(res_paths, lcfg_attrs, mcfg_attrs, ocfg_attrs)

    return res_load


def plot_eigvals_var_L(res_load, naive_ests, oracle_ests, ov_pairs, Ls, K, jind, eigrank, Nnz, gamma_name, savename=None):
    fig, ax = plt.subplots(ov_pairs.shape[0], Ls.size, figsize=(12,8), sharex=True, sharey='row')

    
    for l, L in enumerate(Ls):
        for a, ov_pair in enumerate(ov_pairs):
            ov1 = ov_pair[0]
            ov2 = ov_pair[1]
            lsel = {'L': L, 'gamma': gamma_name}
            msel = {}
            osel = {'ov1': ov1, 'ov2': ov2}
            res = filter_loaded(res_load, lsel, msel, osel)
            if type(res) is list: 
                print('Warning: Using first of multiple results!')
                res = res[0]
            nz_model = res['params']['model_nonzero_inds']
            cfg = res['cfg']

            eigval_true = res['eigvals_true'].squeeze()
            eigvals_em = jnp.stack([x.eigvals.squeeze() for x in res['track']['gamma_lowrank']])

            gamma_naive = naive_ests[l,a,:,:,:]
            gamma_oracle = oracle_ests[l,a,:,:,:]

            oracle_eigval = get_eigval(gamma_oracle[jind,:,:], eigrank)
            naive_eigval = get_eigval(gamma_naive[jind,:,:], eigrank)

            ax[a,l].axhline(eigval_true, linestyle=true_style, color=true_color, linewidth=true_width)
            ax[a,l].axhline(oracle_eigval, linestyle=oracle_style, color=oracle_color, linewidth=oracle_width)
            ax[a,l].axhline(naive_eigval, linestyle=naive_style, color=naive_color, linewidth=naive_width)
            ax[a,l].plot(eigvals_em, color=est_color, linestyle=est_style, linewidth=est_width)

            # ax[a,l].axhline(true_eigval, color='k', label='True')
            # ax[a,l].axhline(naive_eigval, linestyle='--', color=naive_color, label='Naive')
            # ax[a,l].axhline(oracle_eigval, linestyle='--', color='tab:green', label='Oracle',linewidth=3)

            if l == 0:
                ax[a,l].set_ylabel(f'{ov1}e{ov2}')
            if a == 0:
                ax[a,l].set_title(f'L = {L}')
            if a == ov_pairs.shape[0]-1:
                ax[a,l].set_xlabel('EM Iter')

    fig.suptitle(f'{nz_model[jind]+1} Hz ' + rf'$\lambda {eigrank}$')
    plt.tight_layout()

    print('saving plot')
    plt.savefig(savename)

# TODO DRY this func and above
def plot_func_eigvec_var_L(func, funcname, dim, res_load, naive_ests, oracle_ests, ov_pairs, Ls, K, jind, eigrank, Nnz, gamma_name, savename=None):
    fig, ax = plt.subplots(ov_pairs.shape[0], Ls.size, figsize=(12,8), sharex=True, sharey='row')

    
    for l, L in enumerate(Ls):
        for a, ov_pair in enumerate(ov_pairs):
            ov1 = ov_pair[0]
            ov2 = ov_pair[1]
            lsel = {'L': L, 'gamma': gamma_name}
            msel = {}
            osel = {'ov1': ov1, 'ov2': ov2}
            res = filter_loaded(res_load, lsel, msel, osel)
            nz_model = res['params']['model_nonzero_inds']
            cfg = res['cfg']
            print(cfg.latent.L)

            eigvec_true = res['eigvecs_true'][0,:,eigrank-1]
            eigvecs_em = jnp.stack([x.eigvecs[0,:,eigrank-1] for x in res['track']['gamma_lowrank']])
            func_eigvecs_em = jnp.stack([func(x[dim]) for x in eigvecs_em])

            gamma_naive = naive_ests[l,a,:,:,:]
            gamma_oracle = oracle_ests[l,a,:,:,:]

            oracle_eigvec = get_eigvec(gamma_oracle[jind,:,:], eigrank)
            oracle_eigvec = oracle_eigvec*jnp.exp(-1j*jnp.angle(oracle_eigvec[0]))
            naive_eigvec = get_eigvec(gamma_naive[jind,:,:], eigrank)
            naive_eigvec = naive_eigvec*jnp.exp(-1j*jnp.angle(naive_eigvec[0]))

            ax[a,l].axhline(func(eigvec_true[dim]), linestyle=true_style, color=true_color, linewidth=true_width)
            ax[a,l].axhline(func(oracle_eigvec[dim]), linestyle=oracle_style, color=oracle_color, linewidth=oracle_width)
            ax[a,l].axhline(func(naive_eigvec[dim]), linestyle=naive_style, color=naive_color, linewidth=naive_width)
            ax[a,l].plot(func_eigvecs_em, color=est_color, linestyle=est_style, linewidth=est_width)

            # ax[a,l].axhline(true_eigval, color='k', label='True')
            # ax[a,l].axhline(naive_eigval, linestyle='--', color=naive_color, label='Naive')
            # ax[a,l].axhline(oracle_eigval, linestyle='--', color='tab:green', label='Oracle',linewidth=3)

            if l == 0:
                ax[a,l].set_ylabel(f'{ov1}e{ov2}')
            if a == 0:
                ax[a,l].set_title(f'L = {L}')
            if a == ov_pairs.shape[0]-1:
                ax[a,l].set_xlabel('EM Iter')
            if funcname == 'Phase':
                ax[a,l].set_ylim([-jnp.pi,jnp.pi])

    fig.suptitle(f'{nz_model[jind]+1} Hz ' + rf'Eigvec({eigrank}) dim={dim+1} {funcname}')
    plt.tight_layout()

    print('saving plot')
    plt.savefig(savename)



def get_naive_and_oracle(res_load, Ls, ov_pairs, K, Nnz, gamma_name):
    naive_ests = jnp.zeros((Ls.size, ov_pairs.shape[0], Nnz, K, K), dtype=complex)
    oracle_ests = jnp.zeros((Ls.size, ov_pairs.shape[0], Nnz, K, K), dtype=complex)
    for l, L in enumerate(Ls):
        for a, ov_pair in enumerate(ov_pairs):
            ov1 = ov_pair[0]
            ov2 = ov_pair[1]
            print(f'L: {L},  obs_var: {ov1}e{ov2}' )
            lsel = {'L': L,
            'gamma': gamma_name}
            msel = {}
            osel = {'ov1': ov1, 'ov2': ov2}
            res = filter_loaded(res_load, lsel, msel, osel)

            
            if type(res) is list:
                print('Warning: Using first of multiple results!')
                res = res[0]
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


            obs, obs_params = sample_obs(ocfg, xs)
            naive_est = naive_estimator(obs, nz_model)
            naive_ests = naive_ests.at[l,a,:,:,:].set(naive_est)

            gamma_oracle = jnp.einsum('jkl,jil->jkil', zs[nz_model,:,:], zs[nz_model,:,:].conj()).mean(-1)
            oracle_ests = oracle_ests.at[l,a,:,:,:].set(gamma_oracle)
        
    return naive_ests, oracle_ests

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

if __name__=="__main__":
    run()