import itertools
import numpy as np
from scipy.linalg import block_diag

# from cohlib.alg.laplace_gaussian_obs import TrialDataGaussian, GaussianTrial
from cohlib.alg.laplace_sgc import TrialData, SpikeTrialDeltaLogPoisson, SpikeTrialDeltaReLUPoisson, SpikeTrialDeltaIDPoisson, SpikeTrialBernoulli

from cohlib.utils import (
    transform_cov_r2c,
    transform_cov_c2r,
    rearrange_mat,
    reverse_rearrange_mat,
)

# def fit_gaussian_model(data, W, inits, tapers, invQs, etype='approx', num_em_iters=10, max_approx_iters=10, track=False, jax_m_step=False, inverse_correction=False):
def fit_pp_model(
    data, W, inits, tapers, num_em_iters=10, max_approx_iters=10, track=False, inverse_correction=False
):
    # safety / params
    assert isinstance(data, list)
    K = len(data)

    Ls = [data[i].shape[0] for i in range(K)]
    L = Ls[0]

    num_J_vars = W.shape[1]

    # inits
    Gamma_inv_init = inits['Gamma_inv_init']
    params = inits["params"]
    obs_model = inits["obs_model"]
    optim_type = inits["optim_type"]

    track_tapers = []
    Gamma_est_tapers = []
    if tapers is None:
        num_tapers = 1
    else:
        num_tapers = tapers.shape[1]
    for m in range(num_tapers):
        track_taper = []
        if tapers is None:
            taper = None
        else:
            taper = tapers[:,m]
            print(f'Estimating Taper {m+1}')

        for r in range(num_em_iters):
            if tapers is None:
                print(f'EM iter: {r+1}')
            else:
                print(f'EM iter: {r+1} for taper {m+1}')

            if r == 0:
                Gamma_prev_inv = Gamma_inv_init

            mus = np.zeros((L,K*num_J_vars))
            Ups_invs = np.zeros((L,K*num_J_vars,K*num_J_vars))

            for l in range(L):
                # print(f'Laplace Approx trial {l}')
                # trial = get_gaussian_trial_obj(data, invQs, l, W, Gamma_prev_inv, taper=taper)
                trial = get_trial_obj(
                    data, l, W, Gamma_prev_inv, params, taper=taper, obs_model=obs_model, optim_type=optim_type)
                mu, negUps_inv = trial.laplace_approx(max_approx_iters)

                # real reprsentation
                mus[l,:] = mu
                # TODO yikes - make sure laplace_gauss and laplace_spikes match with this stuff
                Ups_invs[l,:,:] = -negUps_inv


            # M-Step
            Gamma_update_complex, Sigma_complex = update_Gamma_complex(mus, Ups_invs, K, num_J_vars)

            if inverse_correction is True:
                Gamma_prev_inv = construct_Gamma_full_real_corrected(Gamma_update_complex, K, num_J_vars, invert=True)
            else:
                Gamma_prev_inv = construct_Gamma_full_real(Gamma_update_complex, K, num_J_vars, invert=True)


            if track is True:
                taper_track_dict = {'gamma_rplus1':Gamma_update_complex, 
                'inv':Gamma_prev_inv, 
                'mus':mus, 
                'Ups_invs': Ups_invs,
                'Sig_complex': Sigma_complex}
                track_taper.append(taper_track_dict)

        track_tapers.append(track_taper)
        Gamma_est_tapers.append(Gamma_update_complex)

    taper_stack = np.stack(Gamma_est_tapers)
    Gamma_est = taper_stack.mean(0)

    if track is True:
        return Gamma_est, Gamma_est_tapers, track_tapers
    else:
        return Gamma_est, Gamma_est_tapers, None
def get_trial_obj(data, l, W, Gamma_inv_prev, params, taper, obs_model, optim_type):
    """
    data is list of spike data (trial x neurons x time)
    """
    if obs_model == "bernoulli":
        SpikeTrial = SpikeTrialBernoulli
    elif obs_model == "poisson-log-delta":
        SpikeTrial = SpikeTrialDeltaLogPoisson
    elif obs_model == "poisson-relu-delta":
        SpikeTrial = SpikeTrialDeltaReLUPoisson
    elif obs_model == "poisson-id-delta":
        SpikeTrial = SpikeTrialDeltaIDPoisson
    else:
        raise ValueError
    trial_data = [group_data[l, :, :] for group_data in data]
    spike_objs = [
        SpikeTrial(data, params[k], taper) for k, data in enumerate(trial_data)
    ]
    trial_obj = TrialData(spike_objs, Gamma_inv_prev, W, params, obs_model, optim_type)
    return trial_obj 

def update_Gamma_complex(mus, Ups_invs, K, num_J_vars):
    """
    M-Step

    mus is (trials x num_J_vars * K)
    Ups_inv is (trials x num_J_vars * K x num_J_vars * K)
    """
    L = mus.shape[0]

    J = int(num_J_vars / 2)
    mus_outer = np.zeros((L, J, K * 2, K * 2))
    Upss = np.zeros((L, J, K * 2, K * 2))

    for l in range(L):
        Ups_inv_j_vecs = get_freq_vecs_real(np.diag(Ups_invs[l, :, :]), K, num_J_vars)
        mu_js = get_freq_vecs_real(mus[l, :], K, num_J_vars)
        for j in range(J):
            mus_outer[l, j, :, :] = np.outer(mu_js[j], mu_js[j])
            Upss[l, j, :, :] = np.diag(1 / Ups_inv_j_vecs[j])

    # enforce circulary symmetry
    k_mask_pre = 1 - np.eye(2)
    k_mask_inv = block_diag(*[k_mask_pre for k in range(K)])
    k_mask = 1 - k_mask_inv

    Gamma_update_complex = np.zeros((J, K, K), dtype=complex)
    Sigmas_complex = np.zeros((L, J, K, K), dtype=complex)
    for l in range(L):
        Sig_real = mus_outer[l, :, :, :] * k_mask + Upss[l, :, :, :]
        Sig_complex = np.zeros((J, K, K), dtype=complex)
        for j in range(J):
            Sig_complex[j, :, :] = transform_cov_r2c(
                rearrange_mat(Sig_real[j, :, :], K)
            )
        Gamma_update_complex += Sig_complex
        Sigmas_complex[l,:,:,:] = Sig_complex

    Gamma_update_complex = Gamma_update_complex / L

    return Gamma_update_complex, Sigmas_complex

def construct_Gamma_full_real(Gamma_update_complex, K, num_J_vars, invert=False, mu_only=False):
    J = int(num_J_vars / 2)
    Gamma_full = np.zeros((K * num_J_vars, K * num_J_vars))
    for j in range(J):
        Gamma_n = Gamma_update_complex[j, :, :]
        if invert is True:
            if mu_only is True:
                Gamma_n = np.linalg.pinv(Gamma_n)
            else:
                Gamma_n = np.linalg.inv(Gamma_n)
        Gamma_n_real = reverse_rearrange_mat(transform_cov_c2r(Gamma_n), K)
        base_filt = np.zeros(num_J_vars)
        j_var = int(j * 2)
        base_filt[j_var : j_var + 2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        # print(j_filt)
        for k in range(K):
            kj = int(k * 2)
            Gamma_full[j_filt, k * num_J_vars + j_var : k * num_J_vars + j_var + 2] = (
                Gamma_n_real[:, kj : kj + 2]
            )

    return Gamma_full

def construct_Gamma_full_real_corrected(Gamma_update_complex, K, num_J_vars, invert=False, mu_only=False):
    J = int(num_J_vars / 2)
    Gamma_full = np.zeros((K * num_J_vars, K * num_J_vars))
    for j in range(J):
        Gamma_n = Gamma_update_complex[j, :, :]
        base_filt = np.zeros(num_J_vars)
        j_var = int(j * 2)
        base_filt[j_var : j_var + 2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        if invert is True:
            Gamma_inv_n = np.linalg.inv(Gamma_n)
            Gamma_inv_n_real = reverse_rearrange_mat(4*transform_cov_c2r(Gamma_inv_n), K)
            # print(j_filt)
            for k in range(K):
                kj = int(k * 2)
                Gamma_full[j_filt, k * num_J_vars + j_var : k * num_J_vars + j_var + 2] = (
                    Gamma_inv_n_real[:, kj : kj + 2]
                )
        else:
            Gamma_n_real = reverse_rearrange_mat(transform_cov_c2r(Gamma_n), K)
            for k in range(K):
                kj = int(k * 2)
                Gamma_full[j_filt, k * num_J_vars + j_var : k * num_J_vars + j_var + 2] = (
                    Gamma_n_real[:, kj : kj + 2]
                )

    return Gamma_full

# def update_Gamma_complex_dc(mus, Ups_invs, K, num_J_vars, dc=True):
#     L = mus.shape[0]
#     J_nodc = int((num_J_vars-1)/2)
#     J = J_nodc 

#     DC_mus_outer = np.zeros((L,K,K))
#     DC_Upss = np.zeros((L,K,K))
#     mus_outer = np.zeros((L,J,K*2,K*2))
#     Upss = np.zeros((L,J,K*2,K*2))

#     for l in range(L):
#         neg_inv_Ups_j_vecs = get_freq_vecs_real_dc(np.diag(Ups_invs[l,:,:]), K, num_J_vars)
#         mu_js = get_freq_vecs_real_dc(mus[l,:], K,num_J_vars)

#         DC_mus_outer[l,:,:] = np.outer(mu_js[0], mu_js[0])
#         DC_Upss[l,:,:] = -np.diag(1/neg_inv_Ups_j_vecs[0])
#         for j in range(J):
#             mus_outer[l,j,:,:] = np.outer(mu_js[j+1], mu_js[j+1])
#             Upss[l,j,:,:] = -np.diag(1/neg_inv_Ups_j_vecs[j+1])

#     # enforce circulary symmetry
#     k_mask_pre = 1 - np.eye(2)
#     k_mask_inv = block_diag(*[k_mask_pre for k in range(K)])
#     k_mask =  1 - k_mask_inv

#     DC_update = np.zeros((K,K))
#     Gamma_update_complex = np.zeros((J,K,K), dtype=complex)
#     for l in range(L):
#         DC = DC_mus_outer[l,:,:] + DC_Upss[l,:,:]
#         Sig_real = mus_outer[l,:,:,:]*k_mask + Upss[l,:,:,:]
#         Sig_complex = np.zeros((J,K,K), dtype=complex)

#         for j in range(J):
#             Sig_complex[j,:,:] = transform_cov_r2c(rearrange_mat(Sig_real[j,:,:],K))

#         DC_update += DC
#         Gamma_update_complex += Sig_complex
#     DC_update = DC_update / L
#     DC_update = np.eye(K)*DC_update

#     # Gamma_update_complex = (1/4)*Gamma_update_complex 

    
#     # prior = np.eye(K) + 0*1j*np.eye(K)
#     # Gamma_update_complex = (L*Gamma_update_complex + prior[None,:,:]) / (2*K + 2 + L - 2*K - 1)
#     Gamma_update_complex = Gamma_update_complex / L

#     return Gamma_update_complex

# def construct_Gamma_full_real_dc(DC_update, Gamma_update_complex, K, num_J_vars, invert=False):
#     J = int((num_J_vars-1)/2)


#     Gamma_full = np.zeros((K*num_J_vars, K*num_J_vars))
#     if invert is True:
#         DC_update = np.linalg.inv(DC_update)
#     base_filt = np.zeros(num_J_vars)
#     base_filt[0] = 1
#     j_filt = np.tile(base_filt.astype(bool), K)

#     for k in range(K):
#         Gamma_full[j_filt,k*num_J_vars] = DC_update[:,k]



#     for j in range(J):
#         Gamma_n = Gamma_update_complex[j,:,:]
#         if invert is True:
#             Gamma_n = np.linalg.inv(Gamma_n)
#         Gamma_n_real = reverse_rearrange_mat(transform_cov_c2r(Gamma_n),K)
#         base_filt = np.zeros(num_J_vars)
#         j_var = int(j*2 + 1)
#         base_filt[j_var:j_var+2] = 1
#         j_filt = np.tile(base_filt.astype(bool), K)
#         # print(j_filt)
#         for k in range(K):
#             kj = int(k*2)
#             Gamma_full[j_filt,k*num_J_vars+j_var:k*num_J_vars+j_var+2] = Gamma_n_real[:,kj:kj+2]

#     return Gamma_full



def get_freq_vecs_real(vec, K,num_J_vars):
    j_vecs = []
    for jv in range(0,num_J_vars,2):
        base_filt = np.zeros(num_J_vars)
        base_filt[jv:jv+2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        vec_j = vec[j_filt]
        j_vecs.append(vec_j)
    return j_vecs

def get_freq_vecs_real_dc(vec, K, num_J_vars):
    j_vecs = []
    base_filt_dc = np.zeros(num_J_vars)
    base_filt_dc[0] = 1
    dc_filt = np.tile(base_filt_dc.astype(bool), K)
    vec_dc = vec[dc_filt]
    # vec_dc_expand = np.concatenate([np.array([vec_dc[k], 0]) for k in range(K)])
    j_vecs.append(vec_dc)
    for jv in range(1,num_J_vars,2):
        base_filt = np.zeros(num_J_vars)
        base_filt[jv:jv+2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        vec_j = vec[j_filt]
        j_vecs.append(vec_j)
    return j_vecs




# Functions 
def all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)
