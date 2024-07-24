import math
import scipy.optimize as op
from scipy.linalg import block_diag
import numpy as np
from abc import ABC, abstractmethod
mvn = np.random.multivariate_normal





import scipy.optimize as op




class PoissonNormalToy():
    def __init__(self, link, mu, sigma2):
        self.link = link
        self.mu = np.array([mu])
        self.sigma2 = np.array([sigma2])
        assert self.mu.ndim == 1
        assert self.sigma2.ndim == 1

        assert link in ['relu', 'log', 'identity']

        if link == 'identity':
            self.cif = cif_idlink
            self.get_optim_funcs = get_optim_funcs_idlink
            self.compute_fisher_info = fisher_info_idlink
            self.m_step = m_step_idlink
        else:
            raise NotImplementedError


    def simulate(self, L, C):
        xs = np.sqrt(self.sigma2)*np.random.randn(L) + self.mu
        lams = self.cif(xs)
        lams_rep = np.tile(lams, C).reshape(C,-1)
        ns = np.random.poisson(lams_rep)
        return xs, lams, ns

    def e_step_optim(self, ns, sigma2, optim_type):
        L = ns.shape[1]
        x_ests = np.zeros(L)
        for l in range(L):
            n = ns[:,l]
            cost_func, cost_grad, cost_hess = self.get_optim_funcs(n, self.mu, sigma2)
            max_iter = 1000

            init = np.array([0.1])

            if optim_type == 'BFGS':
                Result = op.minimize(fun=cost_func, x0=init,
                            jac=cost_grad, method='BFGS',
                            options={'maxiter':max_iter, 'disp':False})
            elif optim_type == 'Newton':
                Result = op.minimize(fun=cost_func, x0=init,
                            jac=cost_grad, hess=cost_hess, method='Newton-CG',
                            options={'maxiter':max_iter, 'disp':False})
            else:
                raise NotImplementedError

            x_ests[l] = Result.x[0]

        x_var_ests = -1/self.compute_fisher_info(x_ests, ns, sigma2)

        return x_ests, x_var_ests

    def estimate_sigma2(self, ns, num_iters, optim_type, track=False, in_place=False, print_iter=None):

        sigma2 = self.sigma2
        L = ns.shape[1]
        if track is True:
            track_sigma2 = np.zeros(num_iters)
            track_x_ests = np.zeros((L, num_iters))
            track_x_var_ests = np.zeros((L, num_iters))
            track_x_sm = np.zeros((L, num_iters))


        for r in range(num_iters):
            if print_iter is None:
                pass
            else:
                if r % print_iter == 0:
                    print(r)
            x_ests, x_var_ests = self.e_step_optim(ns, sigma2, optim_type)
            sigma2 = self.m_step(x_ests, x_var_ests, self.mu)

            if track is True:
                track_sigma2[r] = sigma2
                track_x_ests[:,r] = x_ests
                track_x_var_ests[:,r] = x_var_ests
                track_x_sm[:,r] = x_var_ests + x_ests**2

            if in_place is True:
                self.sigma2 = sigma2

        if track is True:
            track_dict = dict(sigma2=track_sigma2, x_ests=track_x_ests, x_var_ests=track_x_var_ests, x_sm=track_x_sm)
        else:
            track_dict = {}

        results = dict(sigma2_est=sigma2, track=track_dict)
        return results


        


        


        # save init value
        # run EM

        
def cif_idlink(xs):
    return xs

def fisher_info_idlink(x, n, sigma2):
    # hess = -(n/(x**2)) - (1/sigma2)
    A = -(n/(x**2))
    B = -(1/sigma2)
    hess = A.sum() + B
    return hess
    
def m_step_idlink(x_ests, x_var_ests, mu):
    L = x_ests.size
    Ex = x_ests
    Ex2 = x_ests**2 + x_var_ests

    sigma2_est = (1/L)*(Ex2 - 2*mu*Ex + mu**2).sum()
    return sigma2_est

def get_optim_funcs_idlink(n, mu, sigma2):
    def cost_func(x):
        A = n*np.log(x) - x
        B = -( (x-mu)**2 / (2*sigma2) )
        cost = A.sum() + B 
        # cost = n*np.log(x) - x - np.log(math.factorial(n)) - ( (x-mu)**2 / (2*sigma2) )
        # if type(n) == np.ndarray:
        #     if n.ndim == 1:
        #         cost = cost.sum()
        return -cost

    def cost_grad(x):
        A = (n/x) - 1
        B = - ( (x-mu) / (sigma2) )
        grad =  A.sum() + B
        return -grad


    def cost_hess(x):
        A = -(n/(x**2))
        B = -(1/sigma2)
        hess = A.sum() + B
        return -hess

    return cost_func, cost_grad, cost_hess






########
