import argparse

import numpy as np
from scipy.signal.windows import dpss

from cohlib.alg.em_sgc import fit_sgc_model
from cohlib.alg.transform import construct_real_idft_mod

from cohlib.utils import pickle_save, pickle_open, logistic
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

    if args.rho == 0:
        rho = None
        kappa = None
    else: 
        rho = args.rho
        kappa = args.kappa

    load_gamma_path = f'saved/synthetic_data/simple_synthetic_relupoisson_{K}_{L}_{sample_length}_25_0.1_8'
    print(f"Fitting (regularized) poisson data with rho: {rho}, kappa: {kappa}, L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")
    # save_path = f'saved/fitted_models/simple_synthetic_idpoisson_em{num_em}_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}_fitted'
    save_path = f'saved/fitted_models/simple_synthetic_idpoisson_em{num_em}_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}_fitted_test'

    # data_load = pickle_open(load_path)
    gamma_load = pickle_open(load_gamma_path)

    Gamma_true = gamma_load['latent']['Gamma']
    Wv = gamma_load['meta']['Wv']
    fs = gamma_load['meta']['fs']
    freqs = gamma_load['meta']['freqs']

    xs = gamma_load['latent']['xs']
    alphas = np.array([alpha for k in range(K)])

    lams = cif_alpha_id(alphas, xs)
    spikes = sample_spikes_from_xs(lams, C, obs_model='poisson')



    sample_length = spikes.shape[3]
    J_orig = int(sample_length / 2)
    # J_orig = int((Wv.shape[1] - 1) / 2)
    # J_new = J_orig - 451
    J_new = np.where(freqs > 50)[0][0] - 1

    Wv = construct_real_idft_mod(sample_length, J_orig, J_new, fs)
    Wv = Wv[:,1:]

    q = 5
    num_J_vars = Wv.shape[1]
    Gamma_inv_init = np.eye(K*num_J_vars)*q

    # alphas = np.array([alpha for k in range(K)])
    params = [dict(alpha=alpha) for k in range(K)]
    inits = {
        'obs_model': 'poisson-id-delta',
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
    save_dict = dict(spikes_Cavg=spikes.mean(1), lams=lams, Gamma=Gamma_est, tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'])
    pickle_save(save_dict, save_path)

def cif_alpha_relu(alphas, xs):
    lams = alphas[None,:,None] + xs
    lams[lams < 0] = 0
    return lams

def cif_alpha_id(alphas, xs):
    lams = alphas[None,:,None] + xs
    return lams

def Gamma_est_from_zs(zs, dc=True):
    if dc is True:
        zs_outer = np.einsum('ijk,imk->kjmi', zs[:,:,1:], zs[:,:,1:].conj())
    else:
        zs_outer = np.einsum('ijk,imk->kjmi', zs, zs.conj())
    zs_outer_mean = zs_outer.mean(3)
    return zs_outer_mean

if __name__=="__main__":
    run()