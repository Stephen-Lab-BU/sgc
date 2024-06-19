import argparse

import numpy as np
from scipy.signal.windows import dpss

from cohlib.alg.em_sgc_nodc import fit_sgc_model_nodc, construct_Gamma_full_real
from cohlib.alg.transform import construct_real_idft_mod

from cohlib.utils import pickle_save, pickle_open, conv_v_to_z


# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('C', type=int, default=25)
    parser.add_argument('alpha', nargs='?', type=float, default=-3.5)
    parser.add_argument('num_em', nargs='?', type=int, default=10)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    C = args.C
    K = args.K
    alpha = args.alpha
    num_em = args.num_em

    load_path = f'saved/synthetic_data/simple_synthetic_nodc_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}'
    print(f"Fitting Synthetic SGC data with L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")
    save_path = f'saved/fitted_models/simple_synthetic_nodc_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}_fitted'

    data_load = pickle_open(load_path)

    data_load['meta'].keys()
    Gamma_true = data_load['latent']['Gamma']
    Wv = data_load['meta']['Wv']
    fs = data_load['meta']['fs']
    freqs = data_load['meta']['freqs']
    # vs = data_load['latent']['vs']
    zs = data_load['latent']['zs']

    spikes = data_load['observed']['spikes']
    # freqs = data_load['meta']['freqs']

    sample_length = spikes.shape[3]
    J_orig = int(sample_length / 2)
    # J_orig = int((Wv.shape[1] - 1) / 2)
    J_new = np.where(freqs > 50)[0][0] - 1

    Wv = construct_real_idft_mod(sample_length, J_orig, J_new, fs)
     

    q = 5
    num_J_vars = Wv.shape[1]
    J = int(num_J_vars/2)
    Gamma_inv_init = np.eye(K*num_J_vars)*q

    # for k in range(K):
    #     Gamma_inv_init[k,k] = dc_init[k,k]



    Gamma_true_inv = np.stack([np.linalg.inv(Gamma_true[j,:,:]) for j in range(J)])
    # true_init = construct_Gamma_full_real_dc(dc_init, (1/4)*Gamma_true_inv, K, num_J_vars, invert=False)
    # true_init = construct_Gamma_full_real_dc(dc_init, Gamma_true_inv, K, num_J_vars, invert=False)

    # Gamma_est_z = Gamma_est_from_zs(zs)
    # Gamma_sampletrue_inv = np.stack([np.linalg.inv(Gamma_est_z[j,:,:]) for j in range(J)])
    # sampletrue_init = construct_Gamma_full_real_dc(dc_init, Gamma_sampletrue_inv, K, num_J_vars, invert=False)
    
    alphas = np.array([alpha for k in range(K)])
    inits = {
        'Gamma_inv_init': Gamma_inv_init,
        'alphas': alphas,
        'Gamma_true': Gamma_true
        }

    # spikes_short = spikes[:10,:,:,:]
    spikes_use = spikes
    T = spikes.shape[3]
    spikes_grouped = [spikes_use[:,:,k,:] for k in range(K)]

    tapers = None
    # NW = 2
    # Kmax = 3
    # tapers = dpss(sample_length, NW, Kmax).T * 20

    Gamma_est, Gamma_est_tapers, track = fit_sgc_model_nodc(spikes_grouped, Wv, inits, tapers, num_em_iters=num_em, 
                max_approx_iters=50, track=True)

    save_dict = dict(Gamma=Gamma_est, tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'])
    pickle_save(save_dict, save_path)

def Gamma_est_from_zs(zs, dc=True):
    if dc is True:
        zs_outer = np.einsum('ijk,imk->kjmi', zs[:,:,1:], zs[:,:,1:].conj())
    else:
        zs_outer = np.einsum('ijk,imk->kjmi', zs, zs.conj())
    zs_outer_mean = zs_outer.mean(3)
    return zs_outer_mean

if __name__=="__main__":
    run()