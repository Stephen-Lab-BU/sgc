import numpy as np
from cohlib.sample import sample_spikes_from_xs
from cohlib.alg.em_sgc import fit_sgc_model, construct_Gamma_full_real

def cif_alpha_relu(alphas, xs):
    lams = alphas[None,:,None] + xs
    lams[lams < 0] = 0
    return lams

def Gamma_est_from_zs(zs, dc=False):
    if dc is True:
        zs_outer = np.einsum('ijk,imk->kjmi', zs[:,:,1:], zs[:,:,1:].conj())
    else:
        zs_outer = np.einsum('ijk,imk->kjmi', zs, zs.conj())
    zs_outer_mean = zs_outer.mean(3)
    return zs_outer_mean

def load_and_fit_model(Wv, data_load, C, alpha, init_type, optim_type, rho, kappa, store_spikes, num_em, J_model=None, orig_data=None):

    Gamma_true = data_load['latent']['Gamma']
    K = Gamma_true.shape[1]
    fs = data_load['meta']['fs']


    xs = data_load['latent']['xs']
    alphas = np.array([alpha for k in range(K)])

    lams = cif_alpha_relu(alphas, xs)
    spikes = sample_spikes_from_xs(lams, C, delta=1/fs, obs_model='poisson')

    q = 5
    num_J_vars = Wv.shape[1]
    Gamma_inv_init_flat = np.eye(K*num_J_vars)*q

    if J_model is None:
        J_model = int(Wv.shape[1]/2)

    if orig_data is not None:
        zs = orig_data['latent']['zs']
    else:
        zs = data_load['latent']['zs']

    Gamma_est_z = Gamma_est_from_zs(zs)
    Gamma_sampletrue_inv = np.stack([np.linalg.inv(Gamma_est_z[j,:,:]) for j in range(J_model)])
    Gamma_oracle_init = construct_Gamma_full_real(Gamma_sampletrue_inv, K, num_J_vars, invert=False)

    if init_type == 'flat':
        Gamma_inv_init = Gamma_inv_init_flat
    elif init_type == 'oracle':
        Gamma_inv_init = Gamma_oracle_init
    else:
        raise ValueError

    # alphas = np.array([alpha for k in range(K)])
    params = [dict(alpha=alpha) for k in range(K)]
    inits = {
        'obs_model': 'poisson-relu-delta',
        'optim_type': optim_type,
        'Gamma_inv_init': Gamma_inv_init,
        'params':  params,
        'Gamma_true': Gamma_true,
        'rho': rho,
        'kappa': kappa
        }

    spikes_use = spikes
    spikes_grouped = [spikes_use[:,:,k,:] for k in range(K)]

    tapers = None

    Gamma_est, Gamma_est_tapers, track = fit_sgc_model(spikes_grouped, Wv, inits, tapers, num_em_iters=num_em, 
                max_approx_iters=50, track=True)

    # save_dict = dict(Gamma=Gamma_est, tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'], ys=ys)
    if store_spikes is True:
        print('spikes stored')
        save_dict = dict(Gamma=Gamma_est, latent_true=data_load['latent'], spikes=spikes, lams=lams, tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'])
    else:
        save_dict = dict(Gamma=Gamma_est, lams=lams, latent_true=data_load['latent'], tapers=Gamma_est_tapers, Wv=Wv, track=track, inv_init=inits['Gamma_inv_init'])
    return save_dict