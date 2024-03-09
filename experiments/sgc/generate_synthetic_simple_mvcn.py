import argparse
import os

import numpy as np
from cohlib.mvcn import gen_random_mvcn_params, sample_mvcn_time_obs
from cohlib.alg.transform import construct_real_idft
from cohlib.sample import sample_spikes_from_xs

from cohlib.utils import pickle_save, get_dcval, logistic

# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('C', type=int, default=25)
    parser.add_argument('mu', nargs='?', type=float, default=-3.5)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    C = args.C
    K = args.K
    mu = args.mu


    print(f"Generating Synthetic SGC data with L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, mu: {mu}, seed: {seed}")

    save_path = f'saved/synthetic_data/simple_synthetic_{L}_{sample_length}_{C}_{mu}_{seed}'

    fs = 1000
    J = int(sample_length/2)

    dc = np.array([get_dcval(mu, J) for k in range(K)])
    latent, meta = construct_latent_and_sample(sample_length, L, fs, K, dc)
    xs = latent['xs']

    lams = np.apply_along_axis(logistic, 2, xs)
    spikes = sample_spikes_from_xs(lams, C)


    meta['L'] = L
    meta['C'] = C
    meta['seed'] = seed
    meta['sample_length'] = sample_length
    meta['fs'] = 1000


    observed = dict(spikes=spikes, lams=lams, mu=mu)

    save_dict = dict(latent=latent, meta=meta, observed=observed)

    pickle_save(save_dict, save_path)


def construct_latent_and_sample(sample_length, L, fs, K, dc_vals):
    T = sample_length/fs
    Gamma, freqs = gen_random_mvcn_params(T, fs, K)
    n_freqs = Gamma.shape[0]

    # diagonal elements of Gamma for high-coherence frequency
    val1 = 2500
    val2 = 2500

    
    q = 0.1 # scaling for off-diagonal of low-coherence frequencies

    # here we're choosing a single frequency (10) to isolate and set Gamma to have high coherence
    # we let the the rest of the frequencies have low coherence by: 
    low_coh = np.sqrt(q)*np.random.randn(n_freqs-1) + 1j*np.sqrt(q)*np.random.randn(n_freqs-1)

    Gamma[freqs!=10,0,1] = 0.1*low_coh
    Gamma[freqs!=10,1,0] = 0.1*low_coh.conj()

    Gamma[freqs==10,0,0] = val1+0*1j
    Gamma[freqs==10,1,1] = val2+0*1j

    # construct high coherence for target freq
    val = np.sqrt(val1)*np.sqrt(val2)
    init = np.abs(Gamma[freqs==10,1,0])
    scale = val/init
    Gamma[freqs==10,1,0] *= scale*0.92
    Gamma[freqs==10,0,1] = Gamma[freqs==10,1,0].conj()



    # Draw observations from mvcn (in time domain) 

    # TODO refactor for mvcn
    J = freqs.size
    Wv = construct_real_idft(sample_length, J, fs)
    xs, vs, zs = sample_mvcn_time_obs(Gamma, L, freqs, Wv, dc_vals, return_all=True)

    latent = dict(Gamma=Gamma, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs)

    return latent, meta

if __name__ == "__main__":
    run()