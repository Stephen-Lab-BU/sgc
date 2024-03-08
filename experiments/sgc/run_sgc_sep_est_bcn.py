import argparse
import os

import numpy as np
from cohlib.bcn import gen_bcn_params, sample_bcn_time_obs
from numpy.fft import rfft, rfftfreq
from cohlib.estimation import thr_coherence, estimate_coherence
from cohlib.sim import spikes_from_xns
from cohlib.alg.em_lfp import fit_model_lfp

from cohlib.alg.transform import generate_harmonic_dict
from cohlib.utils import conv_real_to_complex

from cohlib.utils import pickle_save, pickle_open


# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    # parser.add_argument('beta_option', type=str, default='betafix', choices=['betafix', 'betafit'])
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
    mu = args.mu
    n_iter = 5

    print(f"Fitting SGC model for synthetic data with L: {L}, sample_length: {sample_length}, C: {C}, seed: {seed}, mu: {mu}")

    load_path = f'saved/synthetic_data/simple_synthetic_{L}_{sample_length}_{C}_{mu}_{seed}'
    save_path = f'saved/fitted_models/em{n_iter}_simple_synthetic_{L}_{sample_length}_{C}_{mu}_{seed}'


    data_load = pickle_open(load_path)

    pp_params = data_load['observed']['pp_params']
    spike_groups = data_load['observed']['spikes']

    sample_length = data_load['meta']['sample_length']
    L = data_load['meta']['L']
    C = data_load['meta']['C']






    # lfp_c = lfp - lfp.mean(1)[:,None]
    # lfp_use = rfft(lfp_c, axis=1, norm='backward')/np.sqrt(sample_length)
    # z_ests_lfp, var_ests_lfp, track_lfp = fit_model_lfp(lfp_use[:,1:], snu_init, num_em_iters=num_em_iters_lfp, EM=True)
    # z_ests_lfp = z_ests_lfp*np.sqrt(sample_length)
    # # TODO rename dict keys in track_lfp
    # snu_est = track_lfp['snu'][-1]
    # sigmas_est = track_lfp['sigmas'][-1,:,:]
    # seps_est = track_lfp['seps'][-1,:]

    # lfp_save_dict = dict(z_ests=z_ests_lfp, var_ests=var_ests_lfp, snu=snu_est, sigmas=sigmas_est, seps=seps_est)

    fs = 1000
    frange = [0, int(fs/2)]
    res = fs/sample_length
    W = generate_harmonic_dict(sample_length, fs, res, frange)

    # TODO add options for fitting params
    # TODO simple method: empirical mean rates - will need this for real data 
    params_init = pp_params

    from cohlib.alg.laplace_spiking import fit_model
    spikes_results = []
    for spike_group in spike_groups:
    # v_ests, var_ests, track = fit_model(spikes, W, params_init, n_iter=n_iter, EM=True)
        v_ests_spikes, var_ests_spikes, track_spikes = fit_model(spike_group, 
                                            W, 
                                            params_init, 
                                            n_iter=n_iter, 
                                            EM=True, 
                                            betafix=True)
        sigma_est = track_spikes['sigmas'][-1,:,:]
        seps_est = track_spikes['seps'][-1,:]

        cnss = []
        for l in range(L):
            cns = conv_zns(v_ests_spikes[l,:][None,:])
            cnss.append(cns)
        z_ests_spikes = np.stack(cnss)
        # plt.plot(freqs[1:], 1000*(cns*cns.conj()).real)
        # plt.xlim([0,100])
        z_ests_spikes = z_ests_spikes*np.sqrt(sample_length*2)


        spikes_group_save_dict = dict(v_ests=v_ests_spikes, var_ests=var_ests_spikes, sigma=sigma_est, seps=seps_est)
        spikes_results.append(spikes_group_save_dict)


    pickle_save(spikes_results, save_path)




def conv_zns(zns):
    L = zns.shape[0]
    J = zns.shape[1]
    znsc = np.zeros((L, int(J/2)), dtype=complex)

    for l in range(L):
        for j in range(int(J/2)):
            ri = j*2
            a = zns[l,ri]
            b = zns[l,ri+1]
            c = conv_real_to_complex(a,b)
            znsc[l,j] = c
    return znsc.squeeze()
    

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