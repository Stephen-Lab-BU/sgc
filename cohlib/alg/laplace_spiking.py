import scipy.optimize as op
import numpy as np
mvn = np.random.multivariate_normal

def laplace_approx(data, W, sigmas, pp_params, max_iter=10):
    """
    Args:
        data: (n_units, n_timepts) array of spiking data
        W: iDFT matrix (or sub-mat)
        proc_var: in theory this should no longer even be here!!!
            - there's an error in the previous model
        pp_params: for future usage, 
    """
    # rint("Running Newton")
    mu = pp_params['mu']
    beta = pp_params['beta']

    C = data.shape[0]
    J = data.shape[1]

    nf = W.shape[1]

    Result = op.minimize(fun=cost_func, x0=np.zeros(nf),
                    args=(data, W, sigmas, pp_params),
                    jac=cost_grad, method='Newton-CG', options={'maxiter':max_iter, 'disp':False})
                    # jac=cost_grad, method='L-BFGS-B', options={'maxiter':2000, 'disp':False})

    z = Result.x

    x = W @ z
    # TODO vectorize this
    H_pre = np.zeros((nf,nf))
    for c in range(C):
        # lamb = 1/(1+np.exp(-x))
        lamb_pre = mu[c] + beta[c] * x
        lamb = 1/(1+np.exp(-lamb_pre))
        H_pre += - W.T @ np.diag(lamb * (1-lamb)) @ W 

    H = H_pre - np.diag(1/sigmas)

    state_variance = np.zeros(nf)
    for i in range(nf):
        state_variance[i] = - (1 / H[i,i])

    return z, state_variance

# TODO double check these 
def cost_func(z, data, W, sigmas, pp_params):
    data = data.astype(bool)
    C = data.shape[0]
    J = data.shape[1]
    mu = pp_params['mu']
    beta = pp_params['beta']

    x = W @ z
    # lamb_pre = x
    lamb_pre = mu[:,None] + beta[:,None] * x
    cost_pre = (data * lamb_pre - np.log(1 + np.exp(lamb_pre)))
    # cost = cost_pre.sum() - ((z)/sigmas).sum()
    cost = cost_pre.sum() - ((z**2)/sigmas).sum()
    # cost = cost_pre.sum() 

    return -cost

def cost_grad(z, data, W, sigmas, pp_params):
    data = data.astype(bool)
    C = data.shape[0]
    J = data.shape[1]
    nf = W.shape[1]
    mu = pp_params['mu']
    beta = pp_params['beta']


    x = W @ z

    lamb = 1/(1+np.exp(-(mu[:,None] + beta[:,None]*x)))
    # lamb = 1/(1+np.exp(-x))
    diff = data - lamb
    g_pre = (beta[None,:] * np.inner(W.T, diff)).sum(1)
    # g_pre = (np.inner(W.T, diff)).sum(1)
    q = (z) / sigmas    
    g = g_pre - q
    # g = g_pre 


    return -g

    
    
    ### 
    # TODO work out math for what we actually want to do here... might need this over trials?
def estimate_pp_params(data, W, zns, zns_var, init_pp_params, m='L-BFGS-B'):
    # zns: (n_trials, nf)
    # zns_var: (n_trials, nf)
    assert data.shape[0] == zns.shape[0]
    assert data.shape[0] == zns_var.shape[0]
    # print(f"Maximizing pp_param with {m}")

    # INITIALIZE PARAMS
    init_params = get_pp_params_vec_from_dict(init_pp_params)
    # print(init_params)

    # what is data shape here? 
    # n_units = data.shape[0]
    # window_size = data.shape[1]
    # n_windows = data.shape[2]

    n_trials = data.shape[0]
    n_units = data.shape[1]
    T = data.shape[2]

    # generate samples from z posterior 
    n_samples = 20
    nf = zns.shape[1]
    # n_windows = state_path.shape[1]

    print('Sampling from zs')
    z_samples = np.zeros((n_trials, nf, n_samples))


    # TODO parallalize?
    for i in range(n_samples):
        if i % 10 == 0:
            print(f'Sampling {i}')
        z_samples[:,:,i] = sample_from_zs(zns, zns_var)
    print('Finished sampling.')

    print('Performing optimization.')
    Result = op.minimize(fun=mc_cost_point_process_vec, x0=init_params,
            args=(data, W, zns, z_samples),
            jac=mc_grad_point_process_vec, method='L-BFGS-B', options={'maxiter':100, 'disp':False})
            # jac=mc_grad_point_process_vec, method=m, options={'maxiter':10, 'xtol':1e-3, 'disp':False})

    pp_params_vec_new = Result.x

    pp_params_new = get_pp_params_dict_from_vec(pp_params_vec_new)
    return pp_params_new

def get_pp_params_dict_from_vec(pp_params_vec):
    n_units = int(pp_params_vec.size / 2)
    
    pp_params = {}
    pp_params['mu'] = pp_params_vec[:n_units]
    pp_params['beta'] = pp_params_vec[n_units:]
    return pp_params

def get_pp_params_vec_from_dict(pp_params_dict):
    mu = pp_params_dict['mu']
    beta = pp_params_dict['beta']

    pp_params_vec = np.concatenate([mu, beta])
    return pp_params_vec

def sample_from_zs(zns, zns_var):
    n_trials = zns.shape[0]
    nf = zns.shape[1]

    sample_list = []
    for l in range(n_trials):
        zn_l = mvn(zns[l,:], np.diag(zns_var[l,:]))
        sample_list.append(zn_l)

    sample = np.stack(sample_list)
        
    return sample
# VECTORIZED
# TODO update for n_windows = 1
def mc_cost_point_process_vec(pp_param_vec, data, W, state_path, state_samples):
    # NOTE I think computationally we can just do a swap axes and everything will be gravy
    # NOTE the reason for this is in our setup, trials are like windows in the state-space setup
    # 
    # TODO fix this up semantically
    # begin swaps
    temp = np.swapaxes(data, 0, 1)
    data = np.swapaxes(temp, 1, 2)

    state_path = np.swapaxes(state_path, 0, 1)
    state_samples = np.swapaxes(state_samples, 0, 1)
    # end swaps

    n_units = data.shape[0]
    n_samples = state_samples.shape[2]
    mu = pp_param_vec[:n_units]
    beta = pp_param_vec[n_units:]

    
    z = state_path
    x = W @ z
    # print(x.shape)

    # TODO remove eventually
    # leaving this here to appreciate the bug D: 
    # A = mu[:,None,None]*(data + beta[:,None,None] * x[None,:])

    # fixed!
    A = data*(mu[:,None,None] + beta[:,None,None] * x[None,:])
    B = 0

    for m in range(n_samples):
        x_sample = W @ state_samples[:,:,m]

        lamb_pre = mu[:,None,None] + beta[:,None,None] * x_sample[None,:]
        B += np.log(1 + np.exp(lamb_pre))

    cost = A.sum() - (1/n_samples)*B.sum()
    
    print(-cost)
    return -cost

    
# TODO update for n_windows -> n_trials
def mc_grad_point_process_vec(pp_param_vec, data, W, state_path, state_samples):
    # TODO see note in cost -- fix this up 
    # begin swaps
    temp = np.swapaxes(data, 0, 1)
    data = np.swapaxes(temp, 1, 2)

    state_path = np.swapaxes(state_path, 0, 1)
    state_samples = np.swapaxes(state_samples, 0, 1)
    # end swaps
    n_units = data.shape[0]

    n_samples = state_samples.shape[2]
    mu = pp_param_vec[:n_units]
    beta = pp_param_vec[n_units:]

    z = state_path
    x = W @ z

    mu_A = data 
    mu_B = 0

    beta_A = data * x[None,:,:]
    beta_B = 0

    for m in range(n_samples):
        x_sample = W @ state_samples[:,:,m]

        lamb_pre = mu[:,None,None] + beta[:,None,None] * x_sample[None,:]
        temp = np.exp(lamb_pre)/(1 + np.exp(lamb_pre))
        mu_B += temp
        beta_B += x_sample * temp

    mu_grad = mu_A.sum((1,2)) - (1/n_samples)*mu_B.sum((1,2))
    beta_grad = beta_A.sum((1,2)) - (1/n_samples)*beta_B.sum((1,2))
    beta_grad = beta_grad 
    
    # TEMP - DELETE THIS FOR MU ESTIMATION, OR CREATE OPTION 
    mu_grad = np.zeros_like(mu)

    vec_grad = np.concatenate([mu_grad, beta_grad])

    return -vec_grad

def get_sample_func(v_ests, var_ests):
    def sample_func(dummy_arg):
        a = v_ests.deepcopy()
        b = var_ests.deepcopy()
        return sample_from_zs(a, b)
    return sample_func


def fit_model(spikes, W, pp_params_init, sigmas=None, n_iter=10, EM=False, max_laplace_iter=2000, betafix=False):
    L = spikes.shape[0]
    nf = W.shape[1]
    if sigmas is None:
        sigmas = 0.1*np.ones((L, nf))
        
    v_ests = np.zeros((L, nf))
    var_ests = np.zeros((L, nf))
#     sigmas = np.zeros((L, sigmas_init.size))
    pp_params = pp_params_init
    pp_params_track = []
    sigmas_track = []
    sigmas_test_track = []
    v_ests_track = []
    var_ests_track = []
    seps_track = []
    if EM: 
        for q in range(n_iter):
            v_ests_new = np.zeros((L, nf))
            var_ests_new = np.zeros((L, nf))
            sigmas_new = np.zeros((L, nf))
            print(f'EM iter {q}')
            for l in range(L):
                # print(f'Trial {l}')
                spikes_l = spikes[l,:,:]
                sigmas_l = sigmas[l,:]
                # print(sigmas_l)
                
                v_est_l, var_est_l = laplace_approx(spikes_l, W, sigmas_l, pp_params, max_iter=max_laplace_iter)
                v_ests_new[l,:] = v_est_l
                var_ests_new[l,:] = var_est_l
                
                sigmas_l_new = v_est_l**2 + var_est_l
                sigmas_new[l,:] = sigmas_l_new

            sigmas_track.append(sigmas_new)
            sigmas_test_track.append(sigmas)
            sigmas = sigmas_new.copy()
            v_ests_track.append(v_ests_new)
            var_ests_track.append(var_ests_new)

            if betafix:
                pp_params = pp_params_init
            else:
                print(f'Estimating pp params on EM {q}')
                pp_params_new = estimate_pp_params(spikes, W, v_ests_new, var_ests_new, pp_params)

                pp_params_track.append(pp_params_new)
                pp_params = pp_params_new


            seps_new = estimate_spec_params(v_ests_new)
            seps_track.append(seps_new)
            sigmas = np.repeat(seps_new[None,:], L, axis=0)
            
        if betafix is False:
            mu_track = [pp_params_track[i]['mu'] for i in range(n_iter)]
            beta_track = [pp_params_track[i]['beta'] for i in range(n_iter)]

        tracking = {}
        if betafix is False:
            tracking['mu'] = np.stack(mu_track)
            tracking['beta'] = np.stack(beta_track)
        # tracking['pp_params'] = pp_params_track
        tracking['sigmas'] = np.stack(sigmas_track)
        tracking['v_ests'] = np.stack(v_ests_track)
        tracking['var_ests'] = np.stack(var_ests_track)
        tracking['seps'] = np.stack(seps_track)

        return v_ests_new, var_ests_new, tracking
    
    else: 
        for l in range(L):
            spikes_l = spikes[l,:,:]
            sigmas_l = sigmas[l,:]
            v_est, var_est = laplace_approx(spikes_l, W, sigmas_l, pp_params, max_iter=max_laplace_iter); 
            v_ests[l,:] = v_est
            var_ests[l,:] = var_est
            
        return v_ests, var_ests

def estimate_spec_params(v_ests):
    L = v_ests.shape[0]

    alpha = 5
    beta = 2
    seps_est_pre = (v_ests**2).sum(0)
    seps_est = (seps_est_pre + beta) / (L + alpha)

    # mean of a_i and b_i for shared variance per freq in CCN
    temp = np.reshape(seps_est, (2, -1), order='F')
    seps_est_pool = np.repeat(temp.mean(0), 2)

    return seps_est_pool