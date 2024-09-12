import scipy.optimize as op
from scipy.linalg import block_diag
import numpy as np
mvn = np.random.multivariate_normal

from cohlib.conv import conv_v_to_z

class TrialDataGaussian():
    def __init__(self, trial_objs, Gamma_inv_prev, W):
        self.trial_objs = trial_objs
        self.W = W
        self.num_J_vars = W.shape[1]
        self.Gamma_inv_prev = Gamma_inv_prev
        self.K = len(trial_objs)

    # TODO
    # - closed form e-step

    def cost_func(self):
        def func(v):
            Jv = self.num_J_vars
            group_terms = np.array(
                [obj.cost_func(v[k*Jv:k*Jv + Jv], self.W) 
                for k, obj in enumerate(self.trial_objs)])
            group_term = group_terms.sum()
            prior_term = (1/2) * (v.T @ self.Gamma_inv_prev @ v)

            cost = group_term - prior_term
            return -cost
        return func
    
    def cost_grad(self):
        def func(v):
            Jv = self.num_J_vars
            group_terms = np.array(
                [obj.cost_grad(v[k*Jv:k*Jv + Jv], self.W) 
                for k, obj in enumerate(self.trial_objs)])
            group_term = group_terms.flatten()
            prior_term = self.Gamma_inv_prev @ v

            grad = group_term - prior_term
            return -grad
        return func

    # TODO change to Hessian for gaussian obs
    def cost_hess(self):
        def func(v):
            Cs = [obj.num_neurons for obj in self.trial_objs]
            Qs = [obj.obs_var_inv for obj in self.trial_objs]
            J = self.num_J_vars

            WQWs = []
            for k in range(self.K):
                WQW = Cs[k] * self.W.T @ Qs[k] @ self.W
                WQWs.append(WQW)
            hess = -block_diag(*WQWs) - self.Gamma_inv_prev


            return -hess
        return func
        # return function of v 

    # TODO update for Gaussian obs
    def laplace_approx(self, max_iter=100):

        cost_func_optim = self.cost_func()
        cost_grad_optim = self.cost_grad()
        cost_hess_optim = self.cost_hess()

        # init = np.zeros(self.num_J_vars*self.K)
        init = np.ones(self.num_J_vars*self.K)
        Result = op.minimize(fun=cost_func_optim, x0=init,
                        # jac=cost_grad_optim, method='BFGS', 
                        jac=cost_grad_optim, hess=cost_hess_optim, method='Newton-CG', 
                        # options={'maxiter':max_iter, 'disp':False})
                        options={'maxiter':2000, 'disp':False})

        mu = Result.x
        neg_invUps = self.compute_Hessian(mu)
        return mu, neg_invUps


    def compute_Hessian(self, v_ests):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        invQs = [obj.obs_var_inv for obj in self.trial_objs]
        J = self.num_J_vars

        WQWs = []
        for k in range(self.K):
            WQW = Cs[k] * self.W.T @ invQs[k] @ self.W
            WQWs.append(WQW)
        Hessian = -block_diag(*WQWs) - self.Gamma_inv_prev


        return Hessian

    def compute_estep_analytical_pinv(self):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        invQs = [obj.obs_var_inv for obj in self.trial_objs]

        pinvW = np.linalg.pinv(self.W)
        yfs = [pinvW @ obj.data_processed for obj in self.trial_objs]
        yf = np.concatenate(yfs)

        invWQWs = []
        for k in range(self.K):
            invWQW = Cs[k] * pinvW @ invQs[k] @ pinvW.T
            invWQWs.append(invWQW)
        CU = block_diag(*invWQWs)
        invUps = CU + self.Gamma_inv_prev

        mu = invUps @ CU @ yf
        neg_invUps = -invUps

        return mu, neg_invUps

    def compute_estep_analytical(self):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        invQs = [obj.obs_var_inv for obj in self.trial_objs]

        yfs = [obj.num_neurons * self.W.T @ obj.obs_var_inv @ obj.data_processed for obj in self.trial_objs]
        yf = np.concatenate(yfs)

        invWQWs = []
        for k in range(self.K):
            invWQW = Cs[k] * self.W.T @ invQs[k] @ self.W
            invWQWs.append(invWQW)
        CU = block_diag(*invWQWs)
        invUps = CU + self.Gamma_inv_prev
        Ups = np.linalg.inv(invUps)

        mu = Ups @ yf
        neg_invUps = -invUps

        return mu, neg_invUps


# NOTE using known observation variance for now
class GaussianTrial():
    def __init__(self, data, obs_var_inv, taper=None):
        self.data = data
        self.num_neurons = data.shape[0]
        self.window_length = data.shape[1]
        self.data_type = 'gaussian'
        self.data_processed = None
        self.obs_var_inv = obs_var_inv


        if taper is None:
            # data_avg = data.astype(int).mean(0)
            data_avg = data.mean(0)
            # self.data_processed = data_avg - data_avg.mean()
            self.data_processed = data_avg 

        else:
            # data_avg = self.data.astype(int).mean(0)
            data_avg = self.data.mean(0)
            data_avg = data_avg - data_avg.mean()
            self.data_processed = self.taper_data(data_avg, taper)

    # TODO reimplement for continuous data
    def taper_data(self, data_avg, taper):
        tapered = data_avg*taper

        return tapered

    def cost_func(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        Q_inv = self.obs_var_inv

        z = conv_v_to_z(v, axis=0)

        x = W @ v

        resid = data - x

        cost = -(1/2)*C*(resid @ Q_inv @ resid)

        return cost

    def cost_grad(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        Q_inv = self.obs_var_inv

        x = W @ v

        diff = (data-x)

        g = C*(W.T @ (Q_inv @ diff))

        return g

import jax.numpy as jnp 
from functools import partial
def add_dc(x, dc):
    dc_arr = jnp.array([dc])
    with_dc = jnp.concatenate([dc_arr, x])
    return with_dc
add0 = partial(add_dc, dc=0)

class GaussianTrialMod():
    """
        Test case - performs similarly to GaussianTrial() )
    """
    def __init__(self, data, obs_var_inv, taper=None):
        self.data = data
        self.num_neurons = data.shape[0]
        self.window_length = data.shape[1]
        self.data_type = 'gaussian'
        self.data_processed = None
        self.obs_var_inv = obs_var_inv


        if taper is None:
            # data_avg = data.astype(int).mean(0)
            data_avg = data.mean(0)
            # self.data_processed = data_avg - data_avg.mean()
            self.data_processed = data_avg 

        else:
            # data_avg = self.data.astype(int).mean(0)
            data_avg = self.data.mean(0)
            data_avg = data_avg - data_avg.mean()
            self.data_processed = self.taper_data(data_avg, taper)

    # TODO reimplement for continuous data
    def taper_data(self, data_avg, taper):
        tapered = data_avg*taper

        return tapered

    def cost_func(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        Q_inv = self.obs_var_inv

        N = int(W.shape[0] / 2)
        Nnz = int(W.shape[1] / 2)
        nz = jnp.arange(Nnz)
        z_full = jnp.zeros(N, dtype=complex)

        z = conv_v_to_z(v, axis=0)
        z_full = z_full.at[nz].set(z)
        z_0dc = jnp.apply_along_axis(add0, 0, z_full)
        x = jnp.fft.irfft(z_0dc, axis=0)

        # x = W @ v

        resid = data - x

        cost = -(1/2)*C*(resid @ Q_inv @ resid)

        return cost

    def cost_grad(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        Q_inv = self.obs_var_inv

        # x = W @ v
        N = int(W.shape[0] / 2)
        Nnz = int(W.shape[1] / 2)
        nz = jnp.arange(Nnz)
        z_full = jnp.zeros(N, dtype=complex)

        z = conv_v_to_z(v, axis=0)
        z_full = z_full.at[nz].set(z)
        z_0dc = jnp.apply_along_axis(add0, 0, z_full)
        x = jnp.fft.irfft(z_0dc, axis=0)

        diff = (data-x)

        g = C*(W.T @ (Q_inv @ diff))

        return g
