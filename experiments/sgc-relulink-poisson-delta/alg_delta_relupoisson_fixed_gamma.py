import argparse

import numpy as np

from cohlib.alg.em_sgc import fit_sgc_model, construct_Gamma_full_real
from cohlib.alg.transform import construct_real_idft_mod

from cohlib.utils import pickle_save, pickle_open
from cohlib.sample import sample_spikes_from_xs


# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('C', type=int, default=25)
    parser.add_argument('alpha', nargs='?', type=float, default=-3.5)
    parser.add_argument('num_em', nargs='?', type=int, default=10)
    parser.add_argument('init_type', nargs='?', type=str, default='flat')
    parser.add_argument('optim_type', nargs='?', type=str, default='BFGS')
    parser.add_argument('rho', nargs='?', type=int, default=0)
    parser.add_argument('kappa', nargs='?', type=int, default=0)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    C = args.C
    K = args.K
    alpha = args.alpha
    num_em = args.num_em
    init_type = args.init_type
    optim_type = args.optim_type

    if args.rho == 0:
        rho = None
        kappa = None
        print(f"Fitting poisson data with ReLU link, L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")
    else: 
        rho = args.rho
        kappa = args.kappa
        print(f"Fitting (regularized) poisson data ReLU link. Params - rho: {rho}, kappa: {kappa}, L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")
    print(f'Using {init_type} init and {optim_type} for optimization.')

    data_path = f'saved/synthetic_data/simple_synthetic_deltarelupoisson_fixed_gamma_{K}_{L}_{sample_length}'
    save_path = f'saved/fitted_models/simple_synthetic_deltarelupoisson_em{num_em}_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}_{init_type}_{optim_type}_fitted'

    # data_load = pickle_open(load_path)
    data_load = pickle_open(data_path)

    Gamma_true = data_load['latent']['Gamma']
    Wv = data_load['meta']['Wv']
    fs = data_load['meta']['fs']
    freqs = data_load['meta']['freqs']

    xs = data_load['latent']['xs']
    alphas = np.array([alpha for k in range(K)])

    lams = cif_alpha_relu(alphas, xs)
    spikes = sample_spikes_from_xs(lams, C, delta=1/fs, obs_model='poisson')



    sample_length = spikes.shape[3]
    J_orig = int(sample_length / 2)
    # J_orig = int((Wv.shape[1] - 1) / 2)
    # J_new = J_orig - 451
    J_new = np.where(freqs > 50)[0][0] - 1

    Wv = construct_real_idft_mod(sample_length, J_orig, J_new, fs)
    Wv = Wv[:,1:]

    q = 5
    num_J_vars = Wv.shape[1]
    Gamma_inv_init_flat = np.eye(K*num_J_vars)*q

    zs = data_load['latent']['zs']
    Gamma_est_z = Gamma_est_from_zs(zs)
    Gamma_sampletrue_inv = np.stack([np.linalg.inv(Gamma_est_z[j,:,:]) for j in range(J_new)])
    Gamma_oracle_init = construct_Gamma_full_real(Gamma_sampletrue_inv, K, num_J_vars, invert=False)

    if init_type == 'flat':
        Gamma_inv_init = Gamma_inv_init_flat
    elif init_type == 'oracle':
        Gamma_inv_init = Gamma_oracle_init
    else:
        raise ValueError

    # alphas = np.array([alpha for k in range(K)])
    params = [dict(alpha=alpha) for k in range(K)]
    inits = {
        'obs_model': 'poisson-relu-delta',
        'optim_type': optim_type,
        'Gamma_inv_init': Gamma_inv_init,
        'params':  params,
        'Gamma_true': Gamma_true,
        'rho': rho,
        'kappa': kappa
        }

    spikes_use = spikes
    spikes_grouped = [spikes_use[:,:,k,:] for k in range(K)]

    tapers = None

    Gamma_est, Gamma_est_tapers, track = fit_sgc_model(spikes_grouped, Wv, inits, tapers, num_em_iters=num_em, 
                max_approx_iters=50, track=True)

    # save_dict = dict(Gamma=Gamma_est, tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'], ys=ys)
    save_dict = dict(Gamma=Gamma_est, lams=lams, tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'])
    pickle_save(save_dict, save_path)

def cif_alpha_relu(alphas, xs):
    lams = alphas[None,:,None] + xs
    lams[lams < 0] = 0
    return lams

def Gamma_est_from_zs(zs, dc=False):
    if dc is True:
        zs_outer = np.einsum('ijk,imk->kjmi', zs[:,:,1:], zs[:,:,1:].conj())
    else:
        zs_outer = np.einsum('ijk,imk->kjmi', zs, zs.conj())
    zs_outer_mean = zs_outer.mean(3)
    return zs_outer_mean

if __name__=="__main__":
    run()