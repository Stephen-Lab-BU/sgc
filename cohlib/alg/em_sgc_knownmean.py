import itertools
import numpy as np
from scipy.linalg import block_diag

from cohlib.alg.laplace_sgc import TrialData_knownmean, SpikeTrial_knownmean

def fit_sgc_model_knownmean(data, W, inits, tapers, num_em_iters=10, max_approx_iters=10, track=False):
    # safety / params
    assert isinstance(data, list)
    K = len(data)

    Ls = [data[i].shape[0] for i in range(K)]
    L = Ls[0]

    num_J_vars = W.shape[1]

    # inits
    Gamma_inv_init = inits['Gamma_inv_init']
    knownmean = inits['knownmean']


    # alg
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
                trial = get_trial_obj_knownmean(data, l, W, Gamma_prev_inv, taper=taper, knownmean=knownmean)
                mu, Ups_inv = trial.laplace_approx(max_approx_iters)

                mus[l,:] = mu
                Ups_invs[l,:,:] = Ups_inv


            # M-Step
            print(f'M-Step for EM iter {r+1}')
            Gamma_update_complex = update_Gamma_complex(mus, Ups_invs, K, num_J_vars)
            Gamma_prev_inv = construct_Gamma_full_real(Gamma_update_complex, K, num_J_vars, invert=True)

            if track is True:
                taper_track_dict = {'gamma':Gamma_update_complex, 'inv':Gamma_prev_inv, 'mus':mus}
                track_taper.append(taper_track_dict)

        track_tapers.append(track_taper)
        Gamma_est_tapers.append(Gamma_update_complex)

    taper_stack = np.stack(Gamma_est_tapers)
    Gamma_est = taper_stack.mean(0)

    if track is True:
        return Gamma_est, Gamma_est_tapers, track_tapers
    else:
        return Gamma_est, Gamma_est_tapers


def get_trial_obj_knownmean(data, l, W, Gamma_inv_prev, taper, knownmean):
    """
    data is list of spike data (trial x neurons x time)
    """
    trial_data = [group_data[l,:,:] for group_data in data]
    spike_objs = [SpikeTrial_knownmean(data, knownmean, taper) for data in trial_data]
    trial_obj = TrialData_knownmean(spike_objs, Gamma_inv_prev, W, knownmean)
    return trial_obj


def transform_cov_c2r(complex_cov):
    dim = complex_cov.shape[0]
    A = np.real(complex_cov)
    B = np.imag(complex_cov)
    rcov = np.zeros((2*dim, 2*dim))
    
    rcov = np.block([[A, -B],
                     [B, A]])

    return rcov/2


def rearrange_mat(mat, K):
    temp = np.tile(np.array([1,0]), K)
    f1 = np.outer(temp, temp)
    f2 = np.roll(f1,1,axis=1)
    f3 = np.roll(f1,1,axis=0)
    f4 = np.roll(f1,1,axis=(0,1))

    A = mat[f1.astype(bool)].reshape(K,-1)
    B = mat[f2.astype(bool)].reshape(K,-1)
    C = mat[f3.astype(bool)].reshape(K,-1)
    D = mat[f4.astype(bool)].reshape(K,-1)
    
    new_mat = np.block([[A, B],
                       [C, D]])
    return new_mat

def reverse_rearrange_mat(mat, K):
    temp = np.tile(np.array([1,0]), K)
    f1 = np.outer(temp, temp)
    f2 = np.roll(f1,1,axis=1)
    f3 = np.roll(f1,1,axis=0)
    f4 = np.roll(f1,1,axis=(0,1))

    dimC = K

    A = mat[:dimC,:dimC]
    B = mat[:dimC,dimC:]
    C = mat[dimC:,:dimC]
    D = mat[dimC:,dimC:]

    new_mat = np.zeros_like(mat)

    new_mat[f1.astype(bool)] = A.flatten()
    new_mat[f2.astype(bool)] = B.flatten()
    new_mat[f3.astype(bool)] = C.flatten()
    new_mat[f4.astype(bool)] = D.flatten()
    
    return new_mat

def est_cov_r2c(real_cov):
    dimR = real_cov.shape[0]
    dimC = int(dimR/2)
    ccov = np.zeros((dimC, dimC), dtype=complex)

    A = real_cov[:dimC,:dimC]
    B = real_cov[:dimC,dimC:]
    C = real_cov[dimC:,:dimC]
    D = real_cov[dimC:,dimC:]

    ccov = (A + D) + 1j*(C - B)

    return ccov

transform_cov_r2c = est_cov_r2c


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


def update_Gamma_complex(mus, Ups_invs, K, num_J_vars):
    '''
    mus is (trials x num_J_vars * K)
    Ups_inv is (trials x num_J_vars * K x num_J_vars * K)
    '''
    L = mus.shape[0]

    J = int(num_J_vars/2)
    mus_outer = np.zeros((L,J,K*2,K*2))
    Upss = np.zeros((L,J,K*2,K*2))
    
    for l in range(L):
        Ups_inv_j_vecs = get_freq_vecs_real(np.diag(Ups_invs[l,:,:]), K, num_J_vars)
        mu_js = get_freq_vecs_real(mus[l,:], K,num_J_vars)
        for j in range(J):
            mus_outer[l,j,:,:] = np.outer(mu_js[j], mu_js[j])
            Upss[l,j,:,:] = -np.diag(1/Ups_inv_j_vecs[j])

    # enforce circulary symmetry
    k_mask_pre = 1 - np.eye(2)
    k_mask_inv = block_diag(*[k_mask_pre for k in range(K)])
    k_mask =  1 - k_mask_inv

    Gamma_update_complex = np.zeros((J,K,K), dtype=complex)
    for l in range(L):
        Sig_real = mus_outer[l,:,:,:]*k_mask + Upss[l,:,:,:]
        Sig_complex = np.zeros((J,K,K), dtype=complex)
        for j in range(J):
            Sig_complex[j,:,:] = est_cov_r2c(rearrange_mat(Sig_real[j,:,:],K))
        Gamma_update_complex += Sig_complex
    Gamma_update_complex = Gamma_update_complex / L

    prior = np.eye(K) + 0*1j*np.eye(K)
    Gamma_update_complex += prior[None,:,:]

    return Gamma_update_complex


def construct_Gamma_full_real(Gamma_update_complex, K, num_J_vars, invert=False):
    J = int(num_J_vars/2)
    Gamma_full = np.zeros((K*num_J_vars, K*num_J_vars))
    for j in range(J):
        Gamma_n = Gamma_update_complex[j,:,:]
        if invert == True:
            Gamma_n = np.linalg.inv(Gamma_n)
        Gamma_n_real = reverse_rearrange_mat(transform_cov_c2r(Gamma_n),K)
        base_filt = np.zeros(num_J_vars)
        j_var = int(j*2)
        base_filt[j_var:j_var+2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        # print(j_filt)
        for k in range(K):
            kj = int(k*2)
            Gamma_full[j_filt,k*num_J_vars+j_var:k*num_J_vars+j_var+2] = Gamma_n_real[:,kj:kj+2]

    return Gamma_full



# Functions 
def all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

# Deprececated Functions



# def get_freq_mats_real(group_block_matrix, K, num_J_vars):
#     """
#     mat is Knum_J_vars*Knum_J_vars 
#     ordered as i = k*num_J_vars + j
#     """
#     j_mats = []
#     for j_var in range(0,num_J_vars,2):
#         base_filt = np.zeros((num_J_vars, num_J_vars))
#         base_filt[j_var:j_var+2,j_var:j_var+2] = 1
#         j_filt = np.tile(base_filt.astype(bool), (K,K))
#         mat_j = group_block_matrix[j_filt].reshape(2*K,-1)
#         j_mats.append(mat_j)
#     return j_mats
# def update_Gamma_real_dc(mus, Ups_invs, K, num_J_vars, dc=True):
#     L = mus.shape[0]
#     if dc is True:
#         J_nodc = int((num_J_vars-1)/2)
#         J = J_nodc 
#     else:
#         # J = int(num_J_vars/2)
#         raise NotImplementedError

#     DC_mus_outer = np.zeros((L,K,K))
#     DC_Upss = np.zeros((L,K,K))
#     mus_outer = np.zeros((L,J,K*2,K*2))
#     Upss = np.zeros((L,J,K*2,K*2))

#     for l in range(L):
#         Ups_inv_j_vecs = get_freq_vecs_real_dc(np.diag(Ups_invs[l,:,:]), K, num_J_vars)
#         mu_js = get_freq_vecs_real_dc(mus[l,:], K,num_J_vars)

#         DC_mus_outer[l,:,:] = np.outer(mu_js[0], mu_js[0])
#         DC_Upss[l,:,:] = -np.diag(1/Ups_inv_j_vecs[0])
#         for j in range(J):
#             mus_outer[l,j,:,:] = np.outer(mu_js[j+1], mu_js[j+1])
#             Upss[l,j,:,:] = -np.diag(1/Ups_inv_j_vecs[j+1])
#     # enforce circulary symmetry
#     k_mask_pre = 1 - np.eye(2)
#     k_mask_inv = block_diag(*[k_mask_pre for k in range(K)])
#     k_mask =  1 - k_mask_inv

#     Gamma_update_real = (mus_outer*k_mask[None,None,:,:]).sum(0) + Upss.sum(0)
#     Gamma_update_real = Gamma_update_real / L

#     DC_update = np.zeros((K,K))
#     for l in range(L):
#         DC = DC_mus_outer[l,:,:] + DC_Upss[l,:,:]
#         DC_update += DC
#     DC_update = DC_update / L
#     DC_update = np.diag(np.diag(DC_update))

#     return DC_update, Gamma_update_real

# def construct_Gamma_full_real_dc_nocomplex(DC_update, Gamma_update_real, K, num_J_vars, Gamma_inv_init, invert=False):
#     J = int((num_J_vars-1)/2)

#     temp = DC_update

#     Gamma_full = np.zeros((K*num_J_vars, K*num_J_vars))
#     if invert is True:
#         if DC_update.sum() != 0:
#             DC_update = np.linalg.inv(DC_update)
#             # DC_update = transform_cov_r2c(np.linalg.inv(transform_cov_c2r(DC_update))).real
#     base_filt = np.zeros(num_J_vars)
#     base_filt[0] = 1
#     j_filt = np.tile(base_filt.astype(bool), K)

#     for k in range(K):
#         Gamma_full[j_filt,k*num_J_vars] = DC_update[:,k]


#     # Gamma_update_real_inv = np.stack([np.linalg.inv(Gamma_update_real[j]) for j in range(J)])

#     for j in range(J):
#         Gamma_j = Gamma_update_real[j,:,:]
#         if invert is True:
#             Gamma_j = np.linalg.inv(Gamma_j)
#         # Gamma_n_real = reverse_rearrange_mat(transform_cov_c2r(Gamma_n),K)
#         base_filt = np.zeros(num_J_vars)
#         j_var = int(j*2 + 1)
#         base_filt[j_var:j_var+2] = 1
#         j_filt = np.tile(base_filt.astype(bool), K)
#         # print(j_filt)
#         for k in range(K):
#             kj = int(k*2)
#             Gamma_full[j_filt,k*num_J_vars+j_var:k*num_J_vars+j_var+2] = Gamma_j[:,kj:kj+2]

#     return Gamma_full
#     return Gamma_inv_init