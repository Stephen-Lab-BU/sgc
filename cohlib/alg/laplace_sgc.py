import scipy.optimize as op
from scipy.linalg import block_diag
import numpy as np
mvn = np.random.multivariate_normal


class TrialData():
    def __init__(self, trial_objs, Gamma_inv_prev, W):
        self.trial_objs = trial_objs
        self.W = W
        self.num_J_vars = W.shape[1]
        self.Gamma_inv_prev = Gamma_inv_prev
        self.K = len(trial_objs)

    def cost_func(self):
        def func(z):
            Jv = self.num_J_vars
            group_terms = np.array(
                [obj.cost_func(z[k*Jv:k*Jv + Jv], self.W) 
                for k, obj in enumerate(self.trial_objs)])
            group_term = group_terms.sum()
            prior_term =  z.T @ self.Gamma_inv_prev @ z 

            cost = group_term - prior_term
            return -cost
        return func
    
    def cost_grad(self):
        def func(z):
            Jv = self.num_J_vars
            group_terms = np.array(
                [obj.cost_grad(z[k*Jv:k*Jv + Jv], self.W) 
                for k, obj in enumerate(self.trial_objs)])
            group_term = group_terms.flatten()
            prior_term = self.Gamma_inv_prev @ z

            grad = group_term - prior_term
            return -grad
        return func

    def cost_hess(self):
        pass
        # return function of z 

    def laplace_approx(self, max_iter=10):

        cost_func_optim = self.cost_func()
        cost_grad_optim = self.cost_grad()

        Result = op.minimize(fun=cost_func_optim, x0=np.zeros(self.num_J_vars*self.K),
                        jac=cost_grad_optim, method='Newton-CG', options={'maxiter':max_iter, 'disp':False})

        mu = Result.x
        Ups_inv = self.compute_Hessian(mu)
        return mu, Ups_inv


    def compute_Hessian(self, z_ests):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        J = self.num_J_vars

        WFkWs = []
        for k in range(self.K):
            z_ests_k = z_ests[k*J:k*J+J]
            inner = -self.W @ z_ests_k
            F_k = np.diag(np.exp(inner) / np.exp(inner)**2)
            WFkW = Cs[k] * self.W @ F_k @ self.W.T
            WFkWs.append(WFkW)
        Hessian = block_diag(*WFkWs) - self.Gamma_inv_prev

        return Hessian


# TODO make ABC for data models
class SpikeTrial():
    def __init__(self, data, mus=None):
        self.data = data
        self.num_neurons = data.shape[0]
        self.window_length = data.shape[1]
        self.data_type = 'spiking'

    def taper():
        pass

    def cost_func(self, z, W, mu=-3.5):
        data = self.data.astype(bool)
        C = self.num_neurons
        mus = mu * np.ones(C) 

        x = W @ z
        lamb_pre = mus[:,None] + x
        cost_pre = (data * lamb_pre - np.log(1 + np.exp(lamb_pre)))
        cost = cost_pre.sum() 

        return cost


    def cost_grad(self, z, W, mu=-3.5):
        data = self.data.astype(bool)
        C = self.num_neurons
        mus = mu * np.ones(C)

        x = W @ z

        lamb = 1/(1+np.exp(-(mus[:,None] + x)))
        # lamb = 1/(1+np.exp(-x))
        diff = data - lamb
        g_pre = (np.inner(W.T, diff)).sum(1)
        # g_pre = (np.inner(W.T, diff)).sum(1)
        g = g_pre 

        return g

    def cif(self, x):
        lamb_pre = x
        return 1 / (1 + np.exp(-lamb_pre))

    def update_mus(self, updated_mus):
        self.mus = updated_mus

