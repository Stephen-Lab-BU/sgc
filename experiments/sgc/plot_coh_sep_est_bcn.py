import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from cohlib.bcn import gen_bcn_params, sample_bcn_time_obs
from numpy.fft import rfft, irfft, rfftfreq
from cohlib.estimation import thr_coherence, estimate_coherence
from cohlib.sim import spikes_from_xns

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

    print(f"Plotting Coherence for fitted model - synthetic data with L: {L}, sample_length: {sample_length}, C: {C}, seed: {seed}, mu: {mu}")

    load_data_path = f'saved/synthetic_data/simple_synthetic_{L}_{sample_length}_{C}_{mu}_{seed}'
    # save_path = f'saved/naive_ests/simple_synthetic_{L}_{sample_length}_{C}_{mu}_{snu}_{seed}'

    load_res_path = f'saved/fitted_models/em{n_iter}_simple_synthetic_{L}_{sample_length}_{C}_{mu}_{seed}'


    data_load = pickle_open(load_data_path)
    freqs = data_load['meta']['freqs']

    results = pickle_open(load_res_path)

    results[0]
    group_z_ests = []
    for result in results:
        v_ests = result['v_ests']
        cnss = []
        for l in range(L):
            cns = conv_zns(v_ests[l,:][None,:])
            cnss.append(cns)
        z_ests_spikes = np.stack(cnss)
        z_ests_spikes = z_ests_spikes*np.sqrt(sample_length*2)
        group_z_ests.append(z_ests_spikes)


    zs_est = np.stack(group_z_ests, axis=1)


    # Gmat = get_Gmat()
    J = zs_est.shape[2]

    cohs = []
    # cohs_se = []
    sample_size = L
    ccov_est = est_Gamma_from_zs(zs_est)
    for j in range(J):
        ccov_est_j = ccov_est[j,:,:].squeeze()
        ccov_est_real = conv_ccov_to_real(ccov_est_j)
        param_vec = vech(ccov_est_real)
        coh_est = calc_coherence(param_vec)

        cohs.append(coh_est)

    #     grad = get_grad_for_ci(param_vec)

    #     inv_ccov_est_real = np.linalg.inv(ccov_est_real)
    #     fi_inv = np.linalg.inv((Gmat.T @ np.kron(inv_ccov_est_real, inv_ccov_est_real) @ Gmat)*(sample_size/2)) # works mvn cov

    #     se = np.sqrt(grad @ fi_inv @ grad)
    #     cohs_se.append(se)

    coh = np.array(cohs)
    # coh_se = np.array(cohs_se)


    from matplotlib import rc
    # rc('font',**{'family':'serif','serif':'Arial'})

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"


    label_size = 12
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(freqs, coh)
    # lb = coh - coh_se*1.96
    # ub = coh + coh_se*1.96
    # ax.fill_between(freqs, lb, ub, color='tab:blue', alpha=0.5)
    # ax.plot(lb)
    # ax.plot(ub)
    ax.set_xlim([0,60])
    ax.set_ylim([-0.1,1.2])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    # ax.set_xticklabels([0,1,5], color='white')
    # ax.set_xticklabels([0,1,5])
    ax.set_xlabel('Freq (Hz)', size=label_size)
    ax.set_ylabel('Coherence', size=label_size)
    ax.set_title('Estimated Spike Spike Coherence', size=label_size+2)
    plt.tight_layout()

    save_path = f'saved/figures/em{n_iter}_simple_synthetic_{L}_{sample_length}_{C}_{mu}_{seed}.png'
    plt.savefig(save_path,dpi=300)
    # save_path = f'saved/figures/simple_synthetic_{L}_{sample_length}_{C}_{mu}_{snu}_{seed}.pdf'
    plt.savefig(save_path)

def est_Gamma_from_zs(zs):
    num_freqs = zs.shape[2]
    num_trials = zs.shape[0]
    Gamma_est = np.zeros((num_freqs, 2, 2), dtype=complex)
    for j in range(num_freqs): 
        zs_j = zs[:,:,j]
        Gamma_j_est = np.zeros((2,2), dtype=complex)
        for l in range(num_trials):
            Gamma_j_est += np.outer(zs_j[l,:], zs_j[l,:].conj())
        Gamma_est[j,:,:] = Gamma_j_est/num_trials
    return Gamma_est

def calc_coherence(param_vec):
    a = param_vec[0]
    b = param_vec[1]
    c = param_vec[6]
    d = param_vec[9]

    return (b**2 + c**2)/(a*d)

def get_grad_for_ci(param_vec):
    a = param_vec[0]
    b = param_vec[1]
    c = param_vec[6]
    d = param_vec[9]

    a_grad = -(b**2 +c**2)/(a**2*d)
    b_grad = 2*b / (a*d)
    c_grad = 2*c / (a*d)
    d_grad = -(b**2 +c**2)/(a*d**2)

    grad = np.zeros_like(param_vec)
    grad[0] = a_grad
    grad[1] = b_grad
    grad[6] = c_grad
    grad[9] = d_grad

    return grad

def vech(mat):
    assert mat.ndim == 2
    vech = []
    a = np.array([mat[0,0]])
    b = mat[:2,1]
    c = mat[:3,2]
    d = mat[:4,3]

    test = np.concatenate([a,b,c,d])
    return test

def conv_ccov_to_real(ccov):
    dim = ccov.shape[0]
    A = np.real(ccov)
    B = np.imag(ccov)
    rcov = np.zeros((2*dim, 2*dim))

    # real components
    rcov[:dim,:dim] = A
    rcov[dim:,dim:] = A

    # imag components
    rcov[:dim,dim:] = B
    rcov[dim:,:dim] = -B
    return 0.5*rcov

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

def get_Gmat():
    Gmat = np.zeros((16,10))
    Gmat[0,0] = 1
    Gmat[1,1] = 1
    Gmat[2,3] = 1
    Gmat[3,6] = 1
    Gmat[4,1] = 1
    Gmat[5,2] = 1
    Gmat[6,4] = 1
    Gmat[7,7] = 1
    Gmat[8,3] = 1
    Gmat[9,4] = 1
    Gmat[10,5] = 1
    Gmat[11,8] = 1
    Gmat[12,6] = 1
    Gmat[13,7] = 1
    Gmat[14,8] = 1
    Gmat[15,9] = 1

    return Gmat

if __name__ == "__main__":
    run()



