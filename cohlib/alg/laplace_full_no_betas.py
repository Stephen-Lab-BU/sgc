import scipy.optimize as op
import numpy as np
from cohlib.utils import conv_real_to_complex, conv_complex_to_real



# let's consider a quick version of this without pp_params
# i think there 
def laplace_approx(data_spikes, data_lfp, W, Gamma, snu, sonly=False, lonly=False):
    """
    Args:
        data: (n_units, n_timepts) array of spiking data
        W: iDFT matrix (or sub-mat)
        proc_var: in theory this should no longer even be here!!!
            - there's an error in the previous model
        pp_params: for future usage, 
    """
    # rint("Running Newton")

    C = data_spikes.shape[0]
    J = data_spikes.shape[1]

    nf = W.shape[1]

    n_z = 2*nf

    # data_lfp = data_lfp/1000

    print('lonly:', lonly)
    Result = op.minimize(fun=cost_func, x0=np.zeros(n_z),
                    args=(data_spikes, data_lfp, W, Gamma, snu, sonly, lonly),
                    jac=cost_grad, method='Newton-CG', options={'maxiter':2000, 'disp':True})
                    # jac=cost_grad, method='L-BFGS-B', options={'maxiter':2000, 'disp':True})
                    # jac=cost_grad, method='BFGS', options={'maxiter':2000, 'disp':True})

    z = Result.x

    state_variance = None

    # TODO finish here when needed 
    # z_n = z[:nf]
    # z_y = z[nf:]

    # x_n = W @ z_n
    # TODO vectorize this
    # H_pre = np.zeros((nf,nf))
    # for c in range(C):
    #     lamb = 1/(1+np.exp(-x_n))
    #     H_pre += - W.T @ np.diag(lamb * (1-lamb)) @ W 

    # H = H_pre - np.diag(1/sigmas)

    # state_variance = np.zeros(nf)
    # for i in range(nf):
    #     state_variance[i] = - (1 / H[i,i])



    return z, state_variance

# TODO double check these 
def cost_func(z, data_spikes, data_lfp, W, Gamma, snu, sonly, lonly):
    C = data_spikes.shape[0]
    J = data_spikes.shape[1]
    nf = int(z.size/2)
    # print(z.size)
    # print(nf)
    T = data_lfp.size

    z_n = z[:nf]
    z_y = z[nf:]

    x_n = W @ z_n
    lamb_pre = x_n
    cost_pre = (data_spikes * lamb_pre - np.log(1 + np.exp(lamb_pre)))
    cost_spikes = cost_pre.sum()

    x_y = W @ z_y
    cost_lfp_pre = (data_lfp - x_y)**2
    # cost_lfp = T*np.log(2*np.pi * snu) + (cost_lfp_pre.sum() / 2*snu)
    cost_lfp = cost_lfp_pre.sum() / 2*snu
    # cost_lfp = cost_lfp_pre.sum() 

    # TEMP
    if sonly:
        cost_lfp = 0
    
    if lonly:
        cost_spikes = 0

    cost_zs = compute_ll3_real_slow(Gamma, z_n, z_y)

    # TODO ignoring covariance for now
    # TODO i think what we'll eventually want to do is:
    # TODO have final term actually do what's in cost func
    # TODO i.e. the REAL version of z_j* @ Gamma_inv_j @ z_j
    cost = cost_spikes - cost_lfp - cost_zs
    # cost = - cost_lfp 
    # cost = cost_pre.sum() 

    return -cost

def cost_grad(z, data_spikes, data_lfp, W, Gamma, snu, sonly, lonly):
    C = data_spikes.shape[0]
    J = data_spikes.shape[1]
    # nf = W.shape[1]
    # print(nf)
    nf = int(z.size/2)

    z_n = z[:nf]
    z_y = z[nf:]

    x_n = W @ z_n
    x_y = W @ z_y

    lamb = 1/(1+np.exp(-x_n))
    diff = data_spikes - lamb
    g_pre_n = (np.inner(W.T, diff)).sum(1)

    # hmm how do we find this?
    diff_y = x_y - data_lfp
    # diff_y = data_lfp - x_y
    # g_pre_y = np.inner(W.T, diff_y) / snu
    g_pre_y = (W.T @ diff_y) / snu

    q = compute_ll3_grad_real(Gamma, z_n, z_y)
    # q_y = (z_y) / sigmas_y
    q_n = q[:nf]
    q_y = q[nf:]

    g_n = g_pre_n - q_n
    g_y = g_pre_y - q_y
    # g_y = g_pre_y

    # TEMP
    if sonly:
        g_y = np.zeros_like(z_y)

    if lonly:
        # print('ping')
        g_n = np.zeros_like(z_n)

    g = np.concatenate([g_n, g_y])
    # print(g)

    return -g

def compute_ll3_real_slow(Gamma, zn_real, zy_real):
    vns_mat = zn_real.reshape(2, -1, order='F')
    zns = conv_real_to_complex(vns_mat[0,:], vns_mat[1,:])
    vys_mat = zy_real.reshape(2, -1, order='F')
    zys = conv_real_to_complex(vys_mat[0,:], vys_mat[1,:])
    
    zs = np.stack([zns, zys])
    
    vals = np.zeros_like(zs[0,:])
    J = vals.size
    for j in range(J):
        vals[j] = zs[:,j].conj().T @ np.linalg.inv(Gamma[j,:,:]) @ zs[:,j]
        
    return vals.real.sum()

# so... gradient is 
def compute_ll3_grad_real(Z, vns, vys):
    vns_mat = vns.reshape(2, -1, order='F')
    zns = conv_real_to_complex(vns_mat[0,:], vns_mat[1,:])
    vys_mat = vys.reshape(2, -1, order='F')
    zys = conv_real_to_complex(vys_mat[0,:], vys_mat[1,:])
    
    zs = np.stack([zns, zys])
    
    J = zs.shape[1]
    grad = np.zeros_like(zs)
    
    for j in range(J):
        Gamma_j = Z[j,:,:]
        z_j = zs[:,j]
        grad[:,j] = Gamma_j @ z_j
        
    a, b = conv_complex_to_real(grad[0,:])
    grad_nr = np.array([a,b]).flatten(order='F')
    
    a, b = conv_complex_to_real(grad[1,:])
    grad_yr = np.array([a,b]).flatten(order='F')
    
    return np.concatenate([grad_nr, grad_yr])

def compute_cost_zs(zn_real, zy_real, Gamma):
    pass