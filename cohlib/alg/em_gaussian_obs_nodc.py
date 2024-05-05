import itertools
import numpy as np
from scipy.linalg import block_diag

from cohlib.alg.laplace_gaussian import TrialDataGaussian, GaussianTrial

from cohlib.utils import est_cov_r2c, transform_cov_c2r, rearrange_mat, reverse_rearrange_mat

def fit_gaussian_model_nodc(data, W, inits, tapers, invQs, etype='approx', num_em_iters=10, max_approx_iters=10, track=False):
    # safety / params
    assert isinstance(data, list)
    K = len(data)

    Ls = [data[i].shape[0] for i in range(K)]
    L = Ls[0]

    num_J_vars = W.shape[1]

    # inits
    Gamma_inv_init = inits['Gamma_inv_init']


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
            neg_invUpss = np.zeros((L,K*num_J_vars,K*num_J_vars))

            for l in range(L):
                # print(f'Laplace Approx trial {l}')
                trial = get_gaussian_trial_obj(data, invQs, l, W, Gamma_prev_inv, taper=taper)
                if etype == 'approx':
                    mu, neg_invUps = trial.laplace_approx(max_approx_iters)
                elif etype == 'analytical':
                    mu, neg_invUps = trial.compute_estep_analytical()
                else:
                    raise ValueError

                # real reprsentation
                mus[l,:] = mu
                neg_invUpss[l,:,:] = neg_invUps


            # M-Step
            print(f'M-Step for EM iter {r+1}')
            Gamma_update_complex = update_Gamma_complex(mus, neg_invUpss, K, num_J_vars)

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
        return Gamma_est, Gamma_est_tapers, None



def get_gaussian_trial_obj(data, invQs, l, W, Gamma_inv_prev, taper):
    """
    data is list of length K 
    each entry is group data (trial x neurons x time)
    """
    trial_data = [group_data[l,:,:] for group_data in data]
    observation_objs = [GaussianTrial(data, invQs[k], taper) for k, data in enumerate(trial_data)]
    trial_obj = TrialDataGaussian(observation_objs, Gamma_inv_prev, W)
    return trial_obj


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


def update_Gamma_complex(mus, neg_invUpss, K, num_J_vars):
    '''
    mus is (trials x num_J_vars * K)
    neg_invUps is (trials x num_J_vars * K x num_J_vars * K)
    '''
    L = mus.shape[0]

    J = int(num_J_vars/2)
    mus_outer = np.zeros((L,J,K*2,K*2))
    Upss = np.zeros((L,J,K*2,K*2))
    
    for l in range(L):
        neg_inv_Ups_j_vecs = get_freq_vecs_real(np.diag(neg_invUpss[l,:,:]), K, num_J_vars)
        mu_js = get_freq_vecs_real(mus[l,:], K,num_J_vars)
        for j in range(J):
            mus_outer[l,j,:,:] = np.outer(mu_js[j], mu_js[j])
            Upss[l,j,:,:] = -np.diag(1/neg_inv_Ups_j_vecs[j])

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
