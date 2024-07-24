import argparse

import numpy as np
from cohlib.mvcn import gen_random_mvcn_params, sample_mvcn_time_obs_nodc
from cohlib.alg.transform import construct_real_idft_mod
from cohlib.sample import sample_spikes_from_xs

from cohlib.utils import pickle_save

# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    parser.add_argument('obs_var1', nargs='?', type=float, default=0)
    parser.add_argument('obs_var2', nargs='?', type=float, default=0)
    parser.add_argument('C', nargs='?', type=int, default=0)
    parser.add_argument('alpha', nargs='?', type=float, default=0)
    args = parser.parse_args()

    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    K = args.K
    C = args.C
    obs_var1 = args.obs_var1
    obs_var2 = args.obs_var2
    alpha = args.alpha

    if C > 0:
        print(f"Generating latent Gaussian data Gaussian observations. L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, obs_var: {obs_var1} x 10^({obs_var2}), seed: {seed}")
        save_path = f'saved/synthetic_data/simple_latent_with_obs_{K}_{L}_{sample_length}_{C}_{alpha}_{obs_var1}_{obs_var2}_{seed}.pkl'

    else:
        print(f"Generating latent Gaussian data. L: {L}, K: {K}, sample_length: {sample_length}, seed: {seed}")
        save_path = f'saved/synthetic_data/simple_latent_{K}_{L}_{sample_length}_{seed}.pkl'

    fs = 1000

    val10 = 200 * fs
    val0 = 1/50 * fs
    # val0 = 5 * fs
    if K == 2:
        # latent, meta = construct_latent_and_sample_bcn(sample_length, L, fs, K)
        latent, meta = construct_latent_and_sample_bcn_mod(sample_length, L, fs, val10, val0)
    else:
        raise NotImplementedError

    if C > 0:
        raise NotImplementedError

    meta['L'] = L
    meta['seed'] = seed
    meta['sample_length'] = sample_length
    meta['fs'] = 1000

    save_dict = dict(latent=latent, meta=meta)

    pickle_save(save_dict, save_path)

# need to add freq_filt adjusts Gamma
def construct_latent_and_sample_bcn_mod(sample_length, L, fs, val10, val0):
    K = 2
    target_freq_hz = 10
    T = sample_length/fs
    Gamma, freqs = gen_random_mvcn_params(T, fs, K)
    scale_duration_multiplier = sample_length / fs

    J = Gamma.shape[0]
    for j in range(J):
        _, eig_vecs = np.linalg.eigh(Gamma[j,:,:])
        mod_vals = np.ones(K)*scale_duration_multiplier #+ np.random.randn(K)
        new_mat = eig_vecs @ np.diag(mod_vals) @ eig_vecs.conj().T
        Gamma[j,:,:] = new_mat

    val1 = val10 * fs
    val2 = val10 * fs
    
    Gamma[freqs==target_freq_hz,0,0] = val1+0*1j
    Gamma[freqs==target_freq_hz,1,1] = val2+0*1j

    # construct high coherence for target freq
    val = np.sqrt(val1)*np.sqrt(val2)
    init = np.abs(Gamma[freqs==10,1,0])
    scale = val/init
    Gamma[freqs==target_freq_hz,1,0] *= scale*0.92
    Gamma[freqs==target_freq_hz,0,1] = Gamma[freqs==target_freq_hz,1,0].conj()

    # reduce off-target power
    Gamma[freqs!=target_freq_hz,:,:] = Gamma[freqs!=target_freq_hz,:,:]*val0

    cutoff_freq = 50
    cutoff_freq_ind = np.where(freqs > cutoff_freq)[0][0]
    Gamma_reduce = Gamma.copy()

    Gamma_reduce = Gamma[:cutoff_freq_ind,:,:]
    freqs_reduce = freqs[:cutoff_freq_ind]
    # Gamma_reduce[cutoff_freq_ind:,:,:] = 0

    # Draw observations from mvcn (in time domain) 
    Wv = construct_real_idft_mod(sample_length, freqs.size, freqs_reduce.size, fs)
    xs, vs, zs = sample_mvcn_time_obs_nodc(Gamma_reduce, L, freqs_reduce, Wv, return_all=True)

    latent = dict(Gamma=Gamma_reduce, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs, freqs_reduce=freqs_reduce, Wv=Wv)

    return latent, meta

if __name__ == "__main__":
    run()

