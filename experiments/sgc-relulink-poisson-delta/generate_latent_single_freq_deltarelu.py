import argparse

import numpy as np
from cohlib.mvcn import gen_random_mvcn_params, sample_mvcn_time_obs_nodc
from cohlib.alg.transform import _construct_real_idft_mod_Jsel
from cohlib.sample import sample_spikes_from_xs

from cohlib.utils import pickle_save, pickle_open

# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('C', type=int, default=25)
    parser.add_argument('alpha', nargs='?', type=float, default=-3.5)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    C = args.C
    K = args.K
    alpha = args.alpha
    J_sel=10


    print(f"Generating Synthetic SGC with single frequency ({J_sel} Hz) Poisson data (no DC) with L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")

    # save_path = f'saved/synthetic_data/simple_synthetic_deltarelupoisson_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}'
    save_path = f'saved/synthetic_data/singlefreq_{J_sel}_latent_deltarelu_{K}_{L}_{sample_length}'
    Gamma_load_path = f'saved/synthetic_data/simple_latent_deltarelu_{K}_{L}_{sample_length}'
    Gamma_load = pickle_open(Gamma_load_path)
    Gamma = Gamma_load['latent']['Gamma']

    fs = 1000

    val10 = 200 * fs
    val0 = 1/50 * fs
    # val0 = 5 * fs
    if K == 2:
        # latent, meta = construct_latent_and_sample_bcn(sample_length, L, fs, K)
        latent, meta = construct_latent_and_sample_bcn_mod(Gamma, sample_length, L, J_sel, fs)
    else:
        raise NotImplementedError


    xs = latent['xs']
    alphas = np.array([alpha for k in range(K)])
    lams = cif_alpha_relu(alphas, xs)
    spikes = sample_spikes_from_xs(lams, C, delta=1/fs, obs_model='poisson')


    meta['L'] = L
    meta['C'] = C
    meta['seed'] = seed
    meta['sample_length'] = sample_length
    meta['fs'] = 1000


    observed = dict(spikes=spikes, lams=lams, alpha=alpha)

    save_dict = dict(latent=latent, meta=meta, observed=observed)

    pickle_save(save_dict, save_path)

def cif_alpha_relu(alphas, xs):
    lams = alphas[None,:,None] + xs
    lams[lams < 0] = 0
    return lams

def construct_latent_and_sample_bcn_mod(Gamma, sample_length, L, J_sel, fs):
    K = 2
    T = sample_length/fs

    _, freqs = gen_random_mvcn_params(T, fs, K)

    Gamma_Jsel = Gamma.copy()
    Gamma_Jsel = Gamma[freqs==J_sel,:,:]
    freqs_Jsel = freqs[freqs==J_sel]
    # Gamma_Jsel[cutoff_freq_ind:,:,:] = 0
    # Gamma_J = Gamma_Jsel

    # Draw observations from mvcn (in time domain) 
    Wv = _construct_real_idft_mod_Jsel(sample_length, freqs.size, list(freqs[freqs==J_sel].astype(int)), fs)
    xs, vs, zs = sample_mvcn_time_obs_nodc(Gamma_Jsel, L, freqs_Jsel, Wv, return_all=True)

    latent = dict(Gamma=Gamma_Jsel, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs, Wv=Wv)

    return latent, meta


if __name__ == "__main__":
    run()

