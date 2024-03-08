import numpy as np

def fit_model_lfp(lfp_f, snu_init, sigmas=None, num_em_iters=10, EM=False):
    L = lfp_f.shape[0]
    nf = lfp_f.shape[1]
    if sigmas is None:
        sigmas = 0.1*np.ones((L, nf))
        
    v_ests = np.zeros((L, nf), dtype=complex)
    var_ests = np.zeros((L, nf))
#     sigmas = np.zeros((L, sigmas_init.size))
    snu = snu_init
    snu_track = []
    seps_track = []
    sigmas_track = []
    sigmas_test_track = []
    v_ests_track = []
    var_ests_track = []
    if EM: 
        for q in range(num_em_iters):
            if q % 10 == 0:
                print(f'EM iter {q}')
            v_ests_new = np.zeros((L, nf), dtype=complex)
            var_ests_new = np.zeros((L, nf))
            sigmas_new = np.zeros((L, nf))
            for l in range(L):
                # print(f'Trial {l}')
                lfp_l = lfp_f[l,:]
                sigmas_l = sigmas[l,:]
                # print(sigmas_l)
                
                # v_est_l, var_est_l = laplace_approx(lfp_l, W, sigmas_l, snu)
                v_est_l, var_est_l = v_update(lfp_l, sigmas_l, snu)
                v_ests_new[l,:] = v_est_l
                var_ests_new[l,:] = var_est_l
                
                # try these two 
                # sigmas_l_new = v_est_l**2 + var_est_l
                sigmas_l_new = var_est_l

                sigmas_new[l,:] = sigmas_l_new

            sigmas_track.append(sigmas_new)
            sigmas_test_track.append(sigmas)
            # sigmas = sigmas_new.copy()
            # sigmas = scipy
            v_ests_track.append(v_ests_new)
            var_ests_track.append(var_ests_new)

            snu_new, seps_new = estimate_params(lfp_f, v_ests_new)

            # seps_new += 1e-12
            snu_track.append(snu_new)
            snu = snu_new.copy()
            seps_track.append(seps_new)
            sigmas = np.repeat(seps_new[None,:], L, axis=0)
            

        tracking = {}
        tracking['snu'] = np.stack(snu_track)
        # tracking['pp_params'] = pp_params_track
        tracking['sigmas'] = np.stack(sigmas_track)
        tracking['snu'] = np.stack(snu_track)
        tracking['seps'] = np.stack(seps_track)
        tracking['v_ests'] = np.stack(v_ests_track)
        tracking['var_ests'] = np.stack(var_ests_track)

        return v_ests_new, var_ests_new, tracking
    
    else: 
        pass
        # for l in range(L):
        #     lfp_l = lfp[l,:,:]
        #     sigmas_l = sigmas[l,:]
        #     v_est, var_est = laplace_approx(lfp_l, W, sigmas_l, snu); 
        #     v_ests[l,:] = v_est
        #     var_ests[l,:] = var_est
            
        return v_ests, var_ests

def v_update(lfp_l, sigma_l, snu):
    J = lfp_l.size
    v_upd = np.zeros(J)
    var_upd = np.zeros(J)
    num = sigma_l * lfp_l
    denom = sigma_l + np.repeat(snu, J)
    v_upd = num / denom
    var_upd = 1 / (1/sigma_l + 1/np.repeat(snu,J))

    return v_upd, var_upd

# def update_sigmas():
#     pass

def estimate_params(lfp_f, v_ests):
    L = lfp_f.shape[0]
    nf = lfp_f.shape[1]

    snu_est = (np.abs(lfp_f - v_ests)**2).mean()
    
    alpha = 1
    beta = 0.5
    seps_est_pre = (np.abs(v_ests)**2).sum(0)
    seps_est = (seps_est_pre + beta) / (L + alpha)

    return snu_est, seps_est
    
    