import scipy.optimize as op
import numpy as np



# let's consider a quick version of this without pp_params
# i think there 
def laplace_approx(data, W, sigmas):
    """
    Args:
        data: (n_units, n_timepts) array of spiking data
        W: iDFT matrix (or sub-mat)
        proc_var: in theory this should no longer even be here!!!
            - there's an error in the previous model
        pp_params: for future usage, 
    """
    # rint("Running Newton")

    C = data.shape[0]
    J = data.shape[1]

    nf = W.shape[1]

    Result = op.minimize(fun=cost_func, x0=np.zeros(nf),
                    args=(data, W, sigmas),
                    jac=cost_grad, method='Newton-CG', options={'maxiter':2000, 'disp':False})
                    # jac=cost_grad, method='L-BFGS-B', options={'maxiter':2000, 'disp':False})

    z = Result.x

    x = W @ z
    # TODO vectorize this
    H_pre = np.zeros((nf,nf))
    for c in range(C):
        lamb = 1/(1+np.exp(-x))
        H_pre += - W.T @ np.diag(lamb * (1-lamb)) @ W 

    H = H_pre - np.diag(1/sigmas)

    state_variance = np.zeros(nf)
    for i in range(nf):
        state_variance[i] = - (1 / H[i,i])

    return z, state_variance

# TODO double check these 
def cost_func(z, data, W, sigmas):
    C = data.shape[0]
    J = data.shape[1]

    x = W @ z
    lamb_pre = x
    cost_pre = (data * lamb_pre - np.log(1 + np.exp(lamb_pre)))
    cost = cost_pre.sum() - ((z)/sigmas).sum()
    # cost = cost_pre.sum() 

    return -cost

def cost_grad(z, data, W, sigmas):
    C = data.shape[0]
    J = data.shape[1]
    nf = W.shape[1]


    x = W @ z

    lamb = 1/(1+np.exp(-x))
    diff = data - lamb
    g_pre = (np.inner(W.T, diff)).sum(1)
    q = (z) / sigmas    
    g = g_pre - q
    g = g_pre 

    return -g