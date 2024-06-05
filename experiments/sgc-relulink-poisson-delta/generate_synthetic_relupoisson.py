import argparse

import numpy as np
from cohlib.mvcn import gen_random_mvcn_params, sample_mvcn_time_obs_nodc
from cohlib.alg.transform import construct_real_idft
from cohlib.sample import sample_spikes_from_xs

from cohlib.utils import pickle_save

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


    print(f"Generating Synthetic SGC Poisson data (no DC) with L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")

    save_path = f'saved/synthetic_data/simple_synthetic_deltarelupoisson_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}'

    fs = 1000

    val10 = 200 * fs
    val0 = 1/50 * fs
    # val0 = 5 * fs
    if K == 2:
        # latent, meta = construct_latent_and_sample_bcn(sample_length, L, fs, K)
        latent, meta = construct_latent_and_sample_bcn_mod(sample_length, L, fs, val10, val0)
    elif K == 3:
        latent, meta = construct_latent_and_sample3(sample_length, L, fs, K)
    # elif K == 15:
    #     latent, meta = construct_latent_and_sample15(sample_length, L, fs, K, alpha)
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

    cutoff_freq = 100
    cutoff_freq_ind = np.where(freqs > cutoff_freq)[0][0]
    Gamma_reduce = Gamma.copy()
    Gamma_reduce[cutoff_freq_ind:,:,:] = 0

    # Draw observations from mvcn (in time domain) 
    Wv = construct_real_idft(sample_length, freqs.size, fs)
    Wv_reduce = Wv[:,1:]
    xs, vs, zs = sample_mvcn_time_obs_nodc(Gamma_reduce, L, freqs, Wv_reduce, return_all=True)

    latent = dict(Gamma=Gamma_reduce, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs, Wv=Wv_reduce)

    return latent, meta

def construct_latent_and_sample3(sample_length, L, fs, K):
    T = sample_length/fs
    Gamma, freqs = gen_random_mvcn_params(T, fs, K)
    n_freqs = Gamma.shape[0]

    scale_duration_multiplier = sample_length / 1000
    J = n_freqs
    for j in range(J):
        _, eig_vecs = np.linalg.eigh(Gamma[j,:,:])
        mod_vals = np.array([1,1,1])*scale_duration_multiplier
        # new_mat = eig_vecs @ np.diag(mod_vals) @ eig_vecs.conj().T
        new_mat = eig_vecs @ np.diag(mod_vals) @ eig_vecs.conj().T
        # new_mat = new_mat*(1e-3*scale_duration_multiplier)
        Gamma[j,:,:] = new_mat

    freq_ind = np.ceil(sample_length / 100).astype(int)
    _, eig_vecs = np.linalg.eigh(Gamma[freq_ind,:,:])

    mod_vals = np.array([15000, 3000, 3000])*scale_duration_multiplier

    Gamma[freq_ind,:,:] = eig_vecs @ np.diag(mod_vals) @ eig_vecs.conj().T

    cutoff_freq = 100
    cutoff_freq_ind = np.where(freqs > cutoff_freq)[0][0]
    # cutoff_multiplier = np.ceil(sample_length / cutoff_freq).astype(int)
    Gamma_reduce = Gamma.copy()
    Gamma_reduce[cutoff_freq_ind:,:,:] = 0
    freqs_reduce = freqs


    # TODO implement and update Wv construction to no DC case
    Wv = construct_real_idft(sample_length, freqs.size, fs)
    Wv_reduce = Wv[:,1:]

    xs, vs, zs = sample_mvcn_time_obs_nodc(Gamma_reduce, L, freqs_reduce, Wv_reduce, return_all=True)

    latent = dict(Gamma=Gamma_reduce, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs, Wv=Wv_reduce)

    return latent, meta


def construct_latent_and_sample_bcn(sample_length, L, fs, K):
    T = sample_length/fs
    Gamma, freqs = gen_random_mvcn_params(T, fs, K)
    n_freqs = Gamma.shape[0]

    val1 = 11 * sample_length
    val2 = 11 * sample_length 
    
    q = 0.1 # scaling for off-diagonal of low-coherence frequencies

    # here we're choosing a single frequency (10) to isolate and set Gamma to have high coherence
    # we let the the rest of the frequencies have low coherence by: 
    low_coh = np.sqrt(q)*np.random.randn(n_freqs-1) + 1j*np.sqrt(q)*np.random.randn(n_freqs-1)

    Gamma[freqs!=10,0,1] = 0.1*low_coh
    Gamma[freqs!=10,1,0] = 0.1*low_coh.conj()

    Gamma[freqs!=10,:,:] *= 2

    Gamma[freqs==10,0,0] = val1+0*1j
    Gamma[freqs==10,1,1] = val2+0*1j

    # construct high coherence for target freq
    val = np.sqrt(val1)*np.sqrt(val2)
    init = np.abs(Gamma[freqs==10,1,0])
    scale = val/init
    Gamma[freqs==10,1,0] *= scale*0.92
    Gamma[freqs==10,0,1] = Gamma[freqs==10,1,0].conj()


    cutoff_freq = 100
    cutoff_freq_ind = np.where(freqs > cutoff_freq)[0][0]
    Gamma_reduce = Gamma.copy()
    Gamma_reduce[cutoff_freq_ind:,:,:] = 0
    freqs_reduce = freqs

    # Draw observations from mvcn (in time domain) 
    Wv = construct_real_idft(sample_length, freqs.size, fs)
    Wv_reduce = Wv[:,1:]

    xs, vs, zs = sample_mvcn_time_obs_nodc(Gamma_reduce, L, freqs_reduce, Wv_reduce, return_all=True)

    latent = dict(Gamma=Gamma_reduce, xs=xs, vs=vs, zs=zs)
    meta = dict(freqs=freqs, Wv=Wv_reduce)

    return latent, meta

if __name__ == "__main__":
    run()

