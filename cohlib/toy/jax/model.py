import math
import scipy.optimize as op

import jax
import jax.numpy as jnp
from efax import BernoulliNP
import jax.random as jr
from jax import grad, hessian
import jax.scipy.optimize as jop

from abc import ABC, abstractmethod


class PoissonNormalToy():
    def __init__(self, link, alpha, sigma2):
        self.link = link
        self.alpha = jnp.array([alpha])
        self.sigma2 = jnp.array([sigma2])
        assert self.alpha.ndim == 1
        assert self.sigma2.ndim == 1

        assert link in ['relu', 'log', 'identity']

        if link == 'identity':
            raise NotImplementedError
            # self.cif = cif_idlink
            # self.get_optim_funcs = get_optim_funcs_idlink
            # self.compute_fisher_info = fisher_info_idlink
            # self.m_step = m_step_idlink
        elif link == 'log':
            self.cif = cif_loglink
            self.get_optim_func = get_optim_func_loglink_jax
            self.m_step = m_step_loglink
        else:
            raise NotImplementedError


    def simulate(self, L):
        rk = jr.key(7)
        xs = jnp.sqrt(self.sigma2)*jr.normal(rk, (L,)) + jnp.array([3])
        # xs = np.sqrt(self.sigma2)*np.random.randn(L) + self.alpha
        lams = self.cif(xs)
        ns = jr.poisson(rk, lams)
        return xs, lams, ns

    def e_step_optim(self, ns, sigma2, optim_type):
        L = ns.size
        x_ests = jnp.zeros(L)
        x_var_ests = jnp.zeros(L)
        init = jnp.array([0.1])
        for l in range(L):
            n = ns[l]
            # cost_func, cost_grad, cost_hess = self.get_optim_funcs(n, self.alpha, sigma2)
            cost_func = self.get_optim_func(n, self.alpha, sigma2)
            cost_func = jax.jit(cost_func)
            cost_grad = grad(cost_func)
            cost_grad = jax.jit(cost_grad)
            cost_hess = hessian(cost_func)
            cost_hess = jax.jit(cost_hess)

            hessian_func = hessian(cost_func)

            max_iter = 10
            if optim_type == 'Newton-manual':
                x_est = init[0].copy()
                for _ in range(max_iter):
                    x_est = x_est - cost_grad(x_est)/cost_hess(x_est)

            elif optim_type == 'BFGS-jaxopt':
                Result = jop.minimize(fun=cost_func, x0=init, method='BFGS',
                                options={'maxiter':max_iter})
                x_est = Result.x[0]

            elif optim_type == 'BFGS-scipy':
                Result = op.minimize(fun=cost_func, x0=init, method='BFGS', 
                                jac=cost_grad, options={'maxiter':max_iter})
                x_est = Result.x[0]

            elif optim_type == 'Newton-scipy':
                Result = op.minimize(fun=cost_func, x0=init, method='Newton-CG', 
                                jac=cost_grad, hess=cost_hess, options={'maxiter':max_iter})
                x_est = Result.x[0]

            x_ests = x_ests.at[l].set(x_est)

            x_var_est = 1/hessian_func(x_ests[l])
            x_var_ests = x_var_ests.at[l].set(x_var_est)

        return x_ests, x_var_ests

    def estimate_sigma2(self, ns, num_iters, optim_type, track=False, in_place=False, print_iter=None):

        sigma2 = self.sigma2.copy()
        L = ns.size
        if track is True:
            track_sigma2 = jnp.zeros(num_iters)
            track_x_ests = jnp.zeros((L, num_iters))
            track_x_var_ests = jnp.zeros((L, num_iters))
            track_x_sm = jnp.zeros((L, num_iters))


        for r in range(num_iters):
            if print_iter is None:
                pass
            else:
                if r % print_iter == 0:
                    print(r)
            x_ests, x_var_ests = self.e_step_optim(ns, sigma2, optim_type)
            sigma2 = self.m_step(x_ests, x_var_ests, self.alpha)

            if track is True:
                track_sigma2 = track_sigma2.at[r].set(sigma2)
                track_x_ests = track_x_ests.at[:,r].set(x_ests)
                track_x_var_ests = track_x_var_ests.at[:,r].set(x_var_ests)
                x_sm = x_var_ests + x_ests**2
                track_x_sm = track_x_sm.at[:,r].set(x_sm)

            if in_place is True:
                self.sigma2 = sigma2

        if track is True:
            track_dict = dict(sigma2=track_sigma2, x_ests=track_x_ests, x_var_ests=track_x_var_ests, x_sm=track_x_sm)
        else:
            track_dict = {}

        results = dict(sigma2_est=sigma2, track=track_dict)
        return results

def get_optim_func_loglink_jax(n, alpha, sigma2):
    def cost_func(x):
        cost = n*x - jnp.exp(x) - ( (x-alpha)**2 / (2*sigma2) )
        return -cost[0]
    return cost_func

def cif_loglink(xs):
    return jnp.exp(xs)
    
def m_step_loglink(x_ests, x_var_ests, alpha):
    L = x_ests.size
    Ex = x_ests
    Ex2 = x_ests**2 + x_var_ests

    sigma2_est = (Ex2 - 2*alpha*Ex + alpha**2).sum() / L
    return sigma2_est



########
