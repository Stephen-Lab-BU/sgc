import argparse
import os

import numpy as np
from cohlib.mvcn import gen_random_mvcn_params, sample_mvcn_time_obs
from cohlib.alg.transform import construct_real_idft, construct_real_idft_mod
from cohlib.sample import sample_spikes_from_xs

from cohlib.utils import pickle_save, pickle_open, get_dcval, logistic

# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('C', type=int, default=25)
    parser.add_argument('mu', nargs='?', type=float, default=-3.5)
    parser.add_argument('obs_var1', nargs='?', type=float, default=5)
    parser.add_argument('obs_var2', nargs='?', type=float, default=3)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    C = args.C
    K = args.K
    mu = args.mu
    ov1 = args.obs_var1
    ov2 = args.obs_var2
    obs_var = ov1 * (10**ov2)

    print(f"Generating Synthetic Gaussian observation data with L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, mu: {mu}, obs_var: {ov1}e{ov2}, seed: {seed}")

    save_path = f'saved/synthetic_data/simple_synthetic_gaussian_{K}_{L}_{sample_length}_{C}_{mu}_{ov1}_{ov2}_{seed}'

    fs = 1000

    latent, meta = construct_latent_and_sample_bcn(sample_length, L, fs, K, mu)
    xs = latent['xs']
    xs_rep = np.stack([xs for _ in range(C)]).swapaxes(0,1)

    ys = xs_rep + np.random.randn(*xs_rep.shape)*np.sqrt(obs_var)


    meta['L'] = L
    meta['C'] = C
    meta['seed'] = seed
    meta['sample_length'] = sample_length
    meta['fs'] = 1000


    observed = dict(ys=ys, var=obs_var)

    save_dict = dict(latent=latent, meta=meta, observed=observed)

    pickle_save(save_dict, save_path)


def construct_latent_and_sample_bcn(sample_length, L, fs, K, mu):
    T = sample_length/fs
    Gamma, freqs = gen_random_mvcn_params(T, fs, K)
    n_freqs = Gamma.shape[0]

    # diagonal elements of Gamma for high-coherence frequency
    # val1 = 1.5 * sample_length
    # val2 = 1.5 * sample_length 

    # val1 = 11 * sample_length
    # val2 = 11 * sample_length 
    
    # q = 0.1 # scaling for off-diagonal of low-coherence frequencies

    # # here we're choosing a single frequency (10) to isolate and set Gamma to have high coherence
    # # we let the the rest of the frequencies have low coherence by: 
    # low_coh = np.sqrt(q)*np.random.randn(n_freqs-1) + 1j*np.sqrt(q)*np.random.randn(n_freqs-1)

    # Gamma[freqs!=10,0,1] = 0.1*low_coh
    # Gamma[freqs!=10,1,0] = 0.1*low_coh.conj()

    # Gamma[freqs!=10,:,:] *= 2

    # Gamma[freqs==10,0,0] = val1+0*1j
    # Gamma[freqs==10,1,1] = val2+0*1j

    # # construct high coherence for target freq
    # val = np.sqrt(val1)*np.sqrt(val2)
    # init = np.abs(Gamma[freqs==10,1,0])
    # scale = val/init
    # Gamma[freqs==10,1,0] *= scale*0.92
    # Gamma[freqs==10,0,1] = Gamma[freqs==10,1,0].conj()

    # # Gamma_reduce = Gamma[:100,:,:]
    # # freqs_reduce = freqs[:100]
    # Gamma_reduce = Gamma.copy()
    # Gamma_reduce[100:,:,:] = 0
    # freqs_reduce = freqs

    load_gamma_path = f'saved/synthetic_data/simple_synthetic_gaussian_2_25_{sample_length}_1_-1.0_1.0_0.0_8'
    gamma_load = pickle_open(load_gamma_path)
    Gamma_reduce = gamma_load['latent']['Gamma']
    freqs_reduce = freqs



    # Draw observations from mvcn (in time domain) 

    # TODO refactor for mvcn
    Wv = construct_real_idft(sample_length, freqs.size, fs)
    Wv_reduce = Wv
    # xs, vs, zs = sample_mvcn_time_obs(Gamma_reduce, L, freqs, Wv, dc_vals, return_all=True)
    # Wv_reduce = construct_real_idft_mod(sample_length, n_freqs, 100, fs)
    # J_mod = Wv_reduce.shape[1]
    J_mod = (sample_length/2)
    dc_vals = np.array([get_dcval(mu, J_mod) for k in range(K)])

    xs, vs, zs = sample_mvcn_time_obs(Gamma_reduce, L, freqs_reduce, Wv_reduce, dc_vals, return_all=True)

    latent = dict(Gamma=Gamma_reduce, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs, Wv=Wv_reduce)

    return latent, meta

if __name__ == "__main__":
    run()

