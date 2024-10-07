import argparse
import os

import numpy as np
from scipy.signal.windows import dpss

from cohlib.alg.em_gaussian_obs import fit_gaussian_model
from cohlib.alg.transform import construct_real_idft_mod

from cohlib.utils import pickle_save, pickle_open, gamma_root

import jax.numpy as jnp
import jax.random as jr
from cohlib.jax.dists import sample_from_gamma
from cohlib.jax.observations import add0



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
    parser.add_argument('num_em', nargs='?', type=int, default=15)
    parser.add_argument('seed', nargs='?', type=int, default=8)
    args = parser.parse_args()

    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    C = args.C
    K = args.K
    mu = args.mu
    num_em = args.num_em
    ov1 = args.obs_var1
    ov2 = args.obs_var2
    obs_var = ov1 * (10**ov2)
    fs = 1e3
    des = 10
    scale_init = (1/(des*2)) 
    etype='analytical'

    lseed = 7
    oseed = 7

    # load_path = f'saved/synthetic_data/simple_synthetic_gaussian_{K}_{L}_{sample_length}_{C}_{mu}_{ov1}_{ov2}_{seed}'
    # load_gamma_path = f'saved/synthetic_data/simple_synthetic_gaussian_{K}_{L}_{sample_length}_1_0.0_1.0_0.0_7'
    print("*TROUBLESHOOTING*")
    print(f"Fitting Synthetic Gaussian observation data with L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, mu: {mu}, obs_var: {ov1}e{ov2}, seed: {seed}")
    save_path = f'saved/fitted_models/corrected_scale_mod_jax_comp_simple_synthetic_gaussian_em{num_em}_{K}_{L}_{sample_length}_{C}_{mu}_{ov1}_{ov2}_{seed}_fitted_{etype}.pkl'

    flow = 1
    fhigh = 50
    sp_target = 5
    sp_offtarget = 1
    gamma_path = os.path.join(gamma_root(), f"k2-full{flow}-{fhigh}-10-{sp_target}-{sp_offtarget}.pickle")
    gamma_load = pickle_open(gamma_path)


    gamma_full = gamma_load['gamma']
    freqs = gamma_load['freqs']
    N = freqs.size
    nz = gamma_load['nonzero_inds']
    nz_target = jnp.array([9])
    K = gamma_full.shape[-1]

    lrk = jr.key(lseed)

    zs = sample_from_gamma(lrk, gamma_full, L)
    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    ork = jr.key(oseed)


    print(f"Sampling observations with variance {ov1}e^{ov2}")
    obs_var = ov1 * 10**ov2
    # obs_var = ocfg.ov1 * 10**{ocfg.ov2}
    jax_ys = xs + jr.normal(ork, xs.shape)*jnp.sqrt(obs_var)
    ys = np.array(jax_ys)[:,None,:,:]
    ys = ys.swapaxes(0,-1)




    sample_length = ys.shape[3]
    J_orig = int(sample_length / 2)
    # J_orig = int((Wv.shape[1] - 1) / 2)
    # J_new = J_orig - 451
    J_new = np.where(freqs > 50)[0][0] 

    Wv_raw = construct_real_idft_mod(sample_length, J_orig, J_new, fs)
    Wv = Wv_raw.copy() *  (1 / (2 * jnp.pi))
     
    num_J_vars = Wv.shape[1]
    Gamma_inv_init = np.eye(K*num_J_vars)*scale_init

    # for k in range(K):
    #     Gamma_inv_init[k,k] = dc_init[k,k]

    inits = {
        'Gamma_inv_init': Gamma_inv_init,
        # 'Gamma_inv_init': true_init,
        # 'Gamma_inv_init': sampletrue_init,
        'mu': mu,
        'Gamma_true': gamma_full
        }

    # spikes_short = spikes[:10,:,:,:]
    ys_use = ys
    ys_grouped = [ys_use[:,:,k,:] for k in range(K)]

    tapers = None
    # NW = 2
    # Kmax = 3
    # tapers = dpss(sample_length, NW, Kmax).T * 20
    invQ = np.diag(np.ones(sample_length)*(1/obs_var))
    invQs = [invQ for k in range(K)]

    # Gamma_est, Gamma_est_tapers, track = fit_gaussian_model(ys_grouped, Wv, inits, tapers, invQs, num_em_iters=num_em, 
    #             max_approx_iters=50, track=True)

    Gamma_est, Gamma_est_tapers, track = fit_gaussian_model(ys_grouped, Wv, inits, tapers, invQs, etype=etype, num_em_iters=num_em, 
                max_approx_iters=0, track=True, inverse_correction=True)

    # save_dict = dict(Gamma=Gamma_est, tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'], ys=ys)
    save_dict = dict(ys_Cavg=ys.mean(1), Gamma=Gamma_est, tapers=Gamma_est_tapers, obs_var=obs_var, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'])
    pickle_save(save_dict, save_path)

def Gamma_est_from_zs(zs, dc=False):
    if dc is True:
        zs_outer = np.einsum('ijk,imk->kjmi', zs[:,:,1:], zs[:,:,1:].conj())
    else:
        zs_outer = np.einsum('ijk,imk->kjmi', zs, zs.conj())
    zs_outer_mean = zs_outer.mean(3)
    return zs_outer_mean

if __name__=="__main__":
    run()