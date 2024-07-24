import argparse

import numpy as np
from cohlib.mvcn import gen_random_mvcn_params, sample_mvcn_time_obs_nodc
from cohlib.alg.transform import _construct_real_idft_mod_Jsel
from cohlib.sample import sample_spikes_from_xs
from cohlib.conv import conv_z_to_v

from cohlib.utils import pickle_save, pickle_open

# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    parser.add_argument('C', nargs='?', type=int, default=0)
    parser.add_argument('alpha', nargs='?', type=float, default=-3.5)
    args = parser.parse_args()


    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    K = args.K
    J_sel=10
    np.random.seed(seed)

    if args.C > 0:
        C = args.C
        alpha = args.alpha
        print(f"Generating latent and spiking data (Poisson-ReLU) using single frequency ({J_sel} Hz) with L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")
        save_path = f'saved/synthetic_data/singlefreq_{J_sel}_deltarelu_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}.pkl'

    else:
        print(f"Generating latent samples with single frequency ({J_sel} Hz) with L: {L}, K: {K}, sample_length: {sample_length}, seed: {seed}")
        save_path = f'saved/synthetic_data/singlefreq_{J_sel}_latent_deltarelu_{K}_{L}_{sample_length}_{seed}.pkl'

    Gamma_load_path = f'saved/synthetic_data/simple_latent_deltarelu_{K}_{L}_{sample_length}_{seed}.pkl'
    Gamma_load = pickle_open(Gamma_load_path)
    Gamma = Gamma_load['latent']['Gamma']
    zs = Gamma_load['latent']['zs']
    freqs = Gamma_load['meta']['freqs']
    freqs_reduce = Gamma_load['meta']['freqs_reduce']

    fs = 1000

    if K == 2:
        latent, meta = construct_latent_and_sample_bcn_mod(Gamma, zs, freqs, freqs_reduce, sample_length, L, J_sel, fs)
    else:
        raise NotImplementedError

    meta['L'] = L
    meta['seed'] = seed
    meta['sample_length'] = sample_length
    meta['fs'] = 1000

    if args.C > 0:
        xs = latent['xs']
        alphas = np.array([alpha for k in range(K)])
        lams = cif_alpha_relu(alphas, xs)
        spikes = sample_spikes_from_xs(lams, C, delta=1/fs, obs_model='poisson')
        observed = dict(spikes=spikes, lams=lams, alpha=alpha)

        meta['C'] = C
        save_dict = dict(latent=latent, meta=meta, observed=observed)

    else:
        save_dict = dict(latent=latent, meta=meta)


    save_dict = dict(latent=latent, meta=meta)

    pickle_save(save_dict, save_path)

def cif_alpha_relu(alphas, xs):
    lams = alphas[None,:,None] + xs
    lams[lams < 0] = 0
    return lams

def construct_latent_and_sample_bcn_mod(Gamma_from_simple, zs_from_simple, freqs, freqs_reduce, sample_length, L, J_sel, fs):
    K = 2
    T = sample_length/fs

    # _, freqs = gen_random_mvcn_params(T, fs, K)


    Gamma_Jsel = Gamma_from_simple.copy()
    Gamma_Jsel = Gamma_from_simple[freqs_reduce==J_sel,:,:]
    freqs_Jsel = freqs_reduce[freqs_reduce==J_sel]
    # Gamma_Jsel[cutoff_freq_ind:,:,:] = 0
    # Gamma_J = Gamma_Jsel
    J = int(sample_length/2)



    # Draw observations from mvcn (in time domain) 
    Wv = _construct_real_idft_mod_Jsel(sample_length, freqs.size, list(freqs[freqs==J_sel].astype(int)), fs)
    # xs, vs, zs = sample_mvcn_time_obs_nodc(Gamma_Jsel, L, freqs_Jsel, Wv, return_all=True)
    zs = zs_from_simple[:,:,9]
    zs = zs[:,:,None]

    vs = conv_z_to_v(zs, axis=2, dc=False)

    xs = np.einsum('ij,abj->abi', Wv, vs)

    latent = dict(Gamma=Gamma_Jsel, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs, Wv=Wv)

    return latent, meta


if __name__ == "__main__":
    run()

