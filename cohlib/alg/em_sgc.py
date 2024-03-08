import scipy.optimize as op
import numpy as np
mvn = np.random.multivariate_normal
import itertools
from cohlib.alg.laplace_sgc import TrialData, SpikeTrial
from scipy.linalg import block_diag

def fit_sgc_model(data, W, inits, num_em_iters=10, max_approx_iters=10, track=False):
    # safety / params
    assert type(data) is list
    K = len(data)

    Ls = [data[i].shape[0] for i in range(K)]
    L = Ls[0]
    # assert all_equal(Ls)

    num_timepts = W.shape[0]
    num_freqs = W.shape[1]

    # inits
    Gamma_inv_init = inits['Gamma_inv_init']

    # alg
    # TODO make classes NOT mutable
    track_gamma = []
    for r in range(num_em_iters):
        print(f'EM iter: {r}')
        if r == 0:
            Gamma_prev_inv = Gamma_inv_init

        mus = np.zeros((L,K*num_freqs))
        Ups_invs = np.zeros((L,K*num_freqs,K*num_freqs))

        for l in range(L):
            print(f'Laplace Approx trial {l}')
            trial = get_trial_obj(data, l, W, Gamma_prev_inv)
            mu, Ups_inv = trial.laplace_approx()
            # real reprsentation
            mus[l,:] = mu
            Ups_invs[l,:,:] = Ups_inv


        # M-Step
        print(f'M-Step for EM iter {r}')
        # Gamma_update_real = update_Gamma_real(mus, Ups_invs, K, num_freqs) 
        Gamma_update_complex = update_Gamma_complex(mus, Ups_invs, K, num_freqs) 

        J = int(num_freqs/2)

        # convert from real to complex
        # conv_Gamma_js = [est_cov_r2c(rearrange_mat(Gamma_update_real[j,:,:],K)) for j in range(J)]
        # Gamma_update_complex = np.stack(conv_Gamma_js)

        # invert and construct full matrix
        Gamma_prev_inv = construct_Gamma_full_real(Gamma_update_complex, K, num_freqs, invert=True)

        if track is True:
            # track_dict = dict(real=Gamma_update_real, complex=Gamma_update_complex, inv=Gamma_prev_inv)
            track_dict = dict(complex=Gamma_update_complex, inv=Gamma_prev_inv)
            track_gamma.append(track_dict)


    if track is True:
        return Gamma_update_complex, Gamma_prev_inv, track_gamma
    else:
        return Gamma_update_complex, Gamma_prev_inv

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

def get_trial_obj(data, l, W, Gamma_inv_prev):
    """
    data is list of spike data (trial x neurons x time)
    """
    trial_data = [group_data[l,:,:] for group_data in data]
    spike_objs = [SpikeTrial(data) for data in trial_data]
    trial_obj = TrialData(spike_objs, Gamma_inv_prev, W)
    return trial_obj

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
            Upss[l,j,:,:] = np.diag(1/Ups_inv_j_vecs[j])

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

    return Gamma_update_complex


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
    


def get_freq_mats_real(group_block_matrix, K, num_J_vars):
    """
    mat is Knum_J_vars*Knum_J_vars 
    ordered as i = k*num_J_vars + j
    """
    j_mats = []
    for j_var in range(0,num_J_vars,2):
        base_filt = np.zeros((num_J_vars, num_J_vars))
        base_filt[j_var:j_var+2,j_var:j_var+2] = 1
        j_filt = np.tile(base_filt.astype(bool), (K,K))
        mat_j = group_block_matrix[j_filt].reshape(2*K,-1)
        j_mats.append(mat_j)
    return j_mats

def get_freq_vecs_real(vec, K,num_J_vars):
    j_vecs = []
    for j in range(0,num_J_vars,2):
        base_filt = np.zeros(num_J_vars)
        base_filt[j:j+2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        vec_j = vec[j_filt]
        j_vecs.append(vec_j)
    return j_vecs



# Functions for complex Gamma (UNFINISHED / UNTESTED)
# TODO get complex version from real mus/ups
def update_Gamma(mus, Ups_invs, K, num_J_vars):
    L = z_ests.shape[0]
    # gamma = np.zeros((num_J_vars,K,K))

    mus_outer = np.zeros((L,num_J_vars,K,K))
    Upss = np.zeros((L,num_J_vars,K,K))
    

    for l in range(L):
        Ups_inv_js = get_freq_mats(Ups_invs[l,:,:], K, num_J_vars)
        mu_js = get_freq_vecs(mus, K,num_J_vars)
        for j in range(num_J_vars):
            mus_outer[l,j,:,:] = np.outer(mu_js[j], mu_js[j])
            Upss[l,j,:,:] = np.linag.inv(Ups_inv_js[j])

    Gamma_update = np.sum([Upss, mus_outer], axis=0)
    return Gamma_update


# TODO fix based on complex/real rep
def get_freq_mats(group_block_matrix, K, num_J_vars):
    """
    mat is Knum_J_vars*Knum_J_vars 
    ordered as i = k*num_J_vars + j
    """
    j_mats = []
    for j in range(num_J_vars):
        base_filt = np.zeros((num_J_vars, num_J_vars))
        base_filt[j,j] = 1
        j_filt = np.tile(base_filt.astype(bool), (K,K))
        mat_j = group_block_matrix[j_filt].reshape(K,-1)
        j_mats.append(mat_j)

    return j_mats



def get_freq_vecs(vec, K,J, conv_to_complex=True):
    j_vecs = []
    for j in range(0,J,2):
        base_filt = np.zeros(J)
        base_filt[j:j+2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        vec_j = vec[j_filt]
        if conv_to_complex:
            vec_j = conv_real_vec_j_to_complex(vec_j, K)
        j_vecs.append(vec_j)
    return j_vecs


from cohlib.utils import conv_complex_to_real
def convert_Gamma_j_complex_to_real(Gamma_j_complex):
    K = Gamma_j_complex.shape[0]
    Gamma_j_real = np.zeros((K*2,K*2))
    for k in range(K):
        c1 
        a1, b1 = conv_complex_to_real(c1)

        for k2 in range(k, K):
            a2, b2 = conv_complex_to_real(c2)

def convert_Ups_inv_js_real_to_complex(Ups_inv_j_real, K):
    Ups_inv_j_complex = np.zeros((K,K), dtype=complex)
    pre = np.diag(Ups_inv_j_real)
    for i, k in enumerate(range(0,K,2)):
        print
        Ups_inv_j_complex[i,i] = pre[k:k+2].sum()/2
    return Ups_inv_j_complex


    #TODO 
    pass

def conv_real_vec_j_to_complex(vec_j, K):
    vec_j_complex = np.zeros(int(K), dtype=complex)
    for i, k in enumerate(range(0,K+1,2)):
        a,b = vec_j[k:k+2] 
        c = conv_real_to_complex(a,b)
        vec_j_complex[i] = c
    return vec_j_complex




def all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

# def update_Gamma_real(mus, Ups_invs, K, num_J_vars):
#     '''
#     mus is (trials x num_J_vars * K)
#     Ups_inv is (trials x num_J_vars * K x num_J_vars * K)
#     '''
#     L = mus.shape[0]

#     J = int(num_J_vars/2)
#     mus_outer = np.zeros((L,J,K*2,K*2))
#     Upss = np.zeros((L,J,K*2,K*2))
    
#     for l in range(L):
#         Ups_inv_j_vecs = get_freq_vecs_real(np.diag(Ups_invs[l,:,:]), K, num_J_vars)
#         mu_js = get_freq_vecs_real(mus[l,:], K,num_J_vars)
#         for j in range(J):
#             mus_outer[l,j,:,:] = np.outer(mu_js[j], mu_js[j])
#             Upss[l,j,:,:] = np.diag(1/Ups_inv_j_vecs[j])

#     Gamma_update = np.zeros((J,K*2,K*2))
#     for l in range(L):
#         Gamma_update += mus_outer[l,:,:,:] + Upss[l,:,:,:]
#     Gamma_update = Gamma_update / L

#     k_mask_pre = 1 - np.eye(2)
#     k_mask_inv = block_diag(*[k_mask_pre for k in range(K)])
#     k_mask =  1 - k_mask_inv

#     for j in range(J):
#         Gamma_update[j,:,:] = Gamma_update[j,:,:]*k_mask

#     return Gamma_update
def update_Gamma_real(mus, Ups_invs, K, num_J_vars):
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
            Upss[l,j,:,:] = np.diag(1/Ups_inv_j_vecs[j])

    k_mask_pre = 1 - np.eye(2)
    k_mask_inv = block_diag(*[k_mask_pre for k in range(K)])
    k_mask =  1 - k_mask_inv

    Gamma_update = np.zeros((J,K*2,K*2))
    for l in range(L):
        Gamma_update += mus_outer[l,:,:,:]*k_mask + Upss[l,:,:,:]
    Gamma_update = Gamma_update / L

    return Gamma_update