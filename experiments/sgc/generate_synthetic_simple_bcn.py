import argparse
import os

import numpy as np
from cohlib.bcn import gen_bcn_params, sample_bcn_time_obs
from numpy.fft import rfft, rfftfreq
from cohlib.estimation import thr_coherence, estimate_coherence
from cohlib.sim import spikes_from_xns

from cohlib.utils import pickle_save


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
    beta = 1

    latent, meta = construct_latent_distribution(sample_length, L, fs)

    for k in range(K):
        pass
    # no edits below

    xns = latent['xns']
    xys = latent['xys']

    meta['L'] = L
    meta['C'] = C
    meta['seed'] = seed
    meta['sample_length'] = sample_length
    meta['fs'] = 1000


    pp_params = get_rep_pp_params(mu, beta, C, swap=False)

    spikes1 = spikes_from_xns(xns, pp_params, L, sample_length)
    spikes2 = spikes_from_xns(xys, pp_params, L, sample_length)
    spikes = [spikes1, spikes2]

    observed = dict(spikes=spikes, pp_params=pp_params, mu=mu, beta=beta)

    save_dict = dict(latent=latent, meta=meta, observed=observed)

    pickle_save(save_dict, save_path)



def construct_latent_distribution(sample_length, L, fs, K=2, norm='backward'):
    T = sample_length/fs
    Gamma, freqs = gen_bcn_params(T)
    n_freqs = Gamma.shape[0]

    # diagonal elements of Gamma for high-coherence frequency
    val1 = 45000
    val2 = 45000

    
    q = 0.1 # scaling for off-diagonal of low-coherence frequencies

    # here we're choosing a single frequency (10) to isolate and set Gamma to have high coherence
    # we let the the rest of the frequencies have low coherence by: 
    low_coh = np.sqrt(q)*np.random.randn(n_freqs-1) + 1j*np.sqrt(q)*np.random.randn(n_freqs-1)

    Gamma[freqs!=10,0,1] = low_coh
    Gamma[freqs!=10,1,0] = low_coh.conj()

    Gamma[freqs!=10,0,1] *= 0.1
    Gamma[freqs!=10,1,0] *= 0.1

    Gamma[freqs==10,0,0] = val1+0*1j
    Gamma[freqs==10,1,1] = val2+0*1j

    val = np.sqrt(val1)*np.sqrt(val2)
    init = np.abs(Gamma[freqs==10,1,0])
    scale = val/init
    Gamma[freqs==10,1,0] *= scale*0.92
    Gamma[freqs==10,0,1] = Gamma[freqs==10,1,0].conj()



    # Draw observations from BVN (in time domain) 

    xns, xys, zs = sample_bcn_time_obs(Gamma, L, return_zs=True, norm=norm)

    xns_c = xns - xns.mean(1)[:,None]
    xys_c = xys - xys.mean(1)[:,None]
    xnfs = rfft(xns_c, axis=1, norm=norm)
    xyfs = rfft(xys_c, axis=1, norm=norm)
    freqs = rfftfreq(sample_length, 1/fs)

    xnfs = xnfs[:,1:]
    xyfs = xyfs[:,1:]
    freqs = freqs[1:]

    coh_true = thr_coherence(Gamma, mag_sq=True)
    coh1 = estimate_coherence(xnfs, xyfs, mag_sq=True)

    z_ind = np.where(freqs==10)[0]

    latent = dict(Gamma=Gamma, xns=xns, xys=xys, xnfs=xnfs, xyfs=xyfs, zs=zs)
    meta = dict(coh_true=coh_true, coh_direct_est=coh1, freqs=freqs)

    return latent, meta

    

    # TODO someway to modulate the data generating procedure (i.e. str of coherence... will have to think about this) 

def get_rep_pp_params(mu, beta, C, swap=False):
    pp_params = {}
    # beta = 0.8


    true_mus = np.repeat(mu, C)
    true_betas = np.repeat(beta, C)

    if swap:
        C2 = int(C/2)
        true_betas[:C2] = -beta

    pp_params['mu'] = true_mus
    pp_params['beta'] = true_betas
    return pp_params

if __name__ == "__main__":
    run()