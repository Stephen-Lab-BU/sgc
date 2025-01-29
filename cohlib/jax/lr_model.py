from abc import abstractmethod, ABC
from functools import partial
from typing_extensions import Protocol

import jax
import jax.random as jr
import jax.numpy as jnp

from cohlib.jax.observations import get_e_step_cost_func
from cohlib.jax.models import LatentFourierModel, JaxOptim
from cohlib.jax.dists import LowRankCCN


def rotate_eigvecs(eigvecs):
    """
    Args:
        eigvecs: J x K x R
    """

    J = eigvecs.shape[0]
    R = eigvecs.shape[2]

    rotated = jnp.zeros_like(eigvecs)

    for j in range(J):
        thetas = jnp.angle(eigvecs[j,0,:])
        rotations = jnp.exp(-1j*thetas)

        rotated = rotated.at[j,:,:].set(eigvecs[j,:,:] * rotations[None,:])
    
    return rotated

def stdize_eigvecs(eigvecs):
    """
    Args:
        eigvecs: J x K x R
    """

    J = eigvecs.shape[0]
    R = eigvecs.shape[2]

    stdized = jnp.zeros_like(eigvecs)

    for j in range(J):
        for r in range(R):
            if eigvecs[j,0,r].real < 0:
                stdized = stdized.at[j,:,r].set(-eigvecs[j,:,r])
            else:
                stdized = stdized.at[j,:,r].set(eigvecs[j,:,r])
    
    return stdized


    # for j in range(J):
    #     for r in range(R):
    #         if eigvecs[j,0,r].real < 0:

    
class LowRankToyModel(LatentFourierModel):
    def __init__(self, track_params=True):
        self.track = {'lrccn': []}
        self.track_lrccn = track_params

    def initialize_latent(self, lrccn):
        self.lrccn = lrccn
        self.Kr = self.lrccn.rank
        self.K = self.lrccn.dim
        self.J = lrccn.Nnz

        self.freqs = lrccn.freqs
        self.nz = lrccn.nz
        self.Nnz = self.freqs.size


    def initialize_observations(self, obs_params, obs_type):
        self.obs_params = obs_params
        self.obs_type = obs_type


    def fit_em(self, data, fit_params):
        
        num_em_iters = fit_params['num_em_iters']
        num_newton_iters = fit_params['num_newton_iters']
        m_step_option = fit_params['m_step_option']
        m_step_params = fit_params.get('m_step_params', {})
        fixed_params = fit_params['fixed_params']

        m_step_params['fixed_params'] = fixed_params


        params = {'obs': self.obs_params,
                  'freqs': self.freqs,
                  'nonzero_inds': self.nz,
                  'K': self.K}

        if m_step_option == 'low-rank-eigh':
            self.m_step = m_step_lowrank_eigh
        else:
            raise NotImplementedError
        self.m_step_params = m_step_params
    
        if self.track_lrccn is True:
            self.track['lrccn'].append(self.lrccn)

        for r in range(num_em_iters):
            self.r = r
            print(f'EM Iter {r+1}')
            gamma_inv = jnp.zeros((self.Nnz, self.K, self.K), dtype=complex)
            gamma_inv_nz = self.lrccn.get_gamma_pinv()
            gamma_inv = gamma_inv.at[self.nz,:,:].set(gamma_inv_nz)
            optimizer = JaxOptim(data, gamma_inv, params, self.obs_type, num_iters=num_newton_iters)
            alphas, Upss = optimizer.run_e_step_par()
            self.alphas = alphas
            self.Upss = Upss

            alphas_outer = jnp.einsum('nkl,nil->nkil', alphas, alphas.conj())


            m_step_params['lrccn_prev'] = self.lrccn
            eigvals_update, eigvecs_update = self.m_step(alphas_outer, 2*Upss, self.m_step_params)
            # TODO review / finalize this
            # NOTE Upsilon is doubled - this empirically matches behavior of implementation using 'real representation'. 
            # Believe the reason is that we are effectively using only 'half' of the variables if considering our optimization 
            # in terms of CR-Calculus (Wirtinger calculus). See "The Complex Gradient Operator and the CR Calculus" (Kreutz-Delgado, 2009)
            # at https://arxiv.org/abs/0906.4835

            eigvecs_update = rotate_eigvecs(eigvecs_update)
            # eigvecs_update = stdize_eigvecs(eigvecs_update)
            print('eigvec update:')
            print(jnp.round(eigvecs_update,2))

            lrccn_update = LowRankCCN(eigvals_update, eigvecs_update, dim=self.K, freqs=self.freqs, nonzero_inds=self.nz)

            if self.track_lrccn is True:
                self.track['lrccn'].append(lrccn_update)
            self.lrccn = lrccn_update

def m_step_lowrank_eigh(alphas_outer, Upss, params):
    lrccn_prev = params['lrccn_prev']
    rank = lrccn_prev.rank
    J = lrccn_prev.Nnz
    fixed_params = params['fixed_params']

    Sigma_ests = (alphas_outer + Upss)
    eigvals_update = jnp.zeros_like(lrccn_prev.eigvals)
    eigvecs_update = jnp.zeros_like(lrccn_prev.eigvecs)

    for j in range(J):
        gamma_ests_j = Sigma_ests[j,:,:,:].mean(-1)
        eigvals_eigh_j, eigvecs_eigh_j = jnp.linalg.eigh(gamma_ests_j)

        if 'eigvals' in fixed_params.keys():
            eigvals_update = fixed_params['eigvals']
        else:
            eigvals_update = eigvals_update.at[j,:rank].set(eigvals_eigh_j[-rank:][::-1])

        if 'eigvecs' in fixed_params.keys():
            eigvecs_update = fixed_params['eigvecs']
        else:
            eigvecs_update = eigvecs_update.at[j,:,:rank].set(eigvecs_eigh_j[:,-rank:][:,::-1])

        return eigvals_update, eigvecs_update


def m_step_lowrank_eigval(alphas_outer, Upss, eigvecs, lrccn_prev):
    Sigma_ests = (alphas_outer + Upss)
    eigvals_update = jnp.zeros_like(lrccn_prev.eigvals)
    J = lrccn_prev.Nnz

    for j in range(J):
        Sigma_ests_j = Sigma_ests[j,:,:,:]
        u = eigvecs[j,:,:].squeeze()

        uuH = jnp.outer(u, u.conj())

        ev_est = jnp.trace(uuH @ Sigma_ests_j.mean(-1)).real
        
        # print(f'NOTE: {u_est.shape}')
        eigvals_update = eigvals_update.at[j,0].set(ev_est)
    
    return eigvals_update, eigvecs

def m_step_lowrank_eigvec(alphas_outer, Upss, eigvals, lrccn_prev, params):
    Sigma_ests = (alphas_outer + Upss)
    J = lrccn_prev.Nnz
    K = lrccn_prev.dim

    eigvecs_update = jnp.zeros_like(lrccn_prev.eigvecs)
    Sigma_ests = (alphas_outer + Upss)

    for j in range(J):
        if params['init_type'] == 'random':
            init_seed = params['m_step_seed']
            u_init = jr.normal(jr.key(j+init_seed), (K,)) + jr.normal(jr.key(j+init_seed+1), (K,))*1j
            u_init = u_init / jnp.linalg.norm(u_init)
        elif params['init_type'] == 'warm_start':
            u_init = lrccn_prev.eigvecs[j,:,0]
        else: 
            raise ValueError

        eigval = eigvals[j,0]
        Sigma_ests_j = Sigma_ests[j,:,:,:]

        u_est = eigvec_optim(eigval, Sigma_ests_j, u_init, ts=False)
        
        # print(f'NOTE: {u_est.shape}')
        eigvecs_update = eigvecs_update.at[j,:,0].set(u_est)
    
    return eigvals, eigvecs_update



def m_step_lowrank_custom(alphas_outer, Upss, params):
    lrccn_prev = params['lrccn_prev']
    rank = lrccn_prev.rank
    J = lrccn_prev.Nnz
    ts_flag = params.get('ts_flag')
    ts_flag2 = params.get('ts_flag2')

    fixed_u_mods = ['fixed_u_true', 'fixed_u_oracle']
    fixed_eigval_mods = ['fixed_eigval_true', 'fixed_eigval_oracle']

    if rank != 1:
        raise NotImplementedError


    if ts_flag in fixed_u_mods:
        print(f'M-Step: Estimating eigval; eigvec held using {ts_flag}')
        eigvecs = params['u']
        return m_step_lowrank_eigval(alphas_outer, Upss, eigvecs, lrccn_prev)

    elif ts_flag in fixed_eigval_mods:
        eigvals = params['eigvals']
        print(f'M-Step: Estimating eigvec; eigval held using {ts_flag}')
        return m_step_lowrank_eigvec(alphas_outer, Upss, eigvals, lrccn_prev, params)

    else:

        # Sigma_ests = (alphas_outer + Upss)

        # # if ts_flag == 'mstep_init_prev':
        # #     u_init 
        # u_init = jr.normal(jr.key(init_seed), (K,)) + jr.normal(jr.key(init_seed+1), (K,))*1j
        # u_init = u_init / jnp.linalg.norm(u_init)
        # u_est = u_init
        # # print(f'INIT SHAPE: {u_init.shape}')

        # eigvals_update = jnp.zeros_like(lrccn_prev.eigvals)
        # eigvecs_update = jnp.zeros_like(lrccn_prev.eigvecs)

        # M = 5
        # for j in range(J):
        #     Sigma_ests_j = Sigma_ests[j,:,:,:]
        #     for m in range(M):
        #         uuH = jnp.outer(u_est, u_est.conj())

        #         ev_est = jnp.trace(uuH @ Sigma_ests_j.mean(-1)).real

        #         u_est = eigvec_optim(ev_est, Sigma_ests_j, u_est, ts=False)
            
        #     # print(f'NOTE: {u_est.shape}')
        #     eigvals_update = eigvals_update.at[j,0].set(ev_est)
        #     eigvecs_update = eigvecs_update.at[j,:,0].set(u_est)
        
        # return eigvals_update, eigvecs_update
        
        # TODO update to use m_step_lowrank_eigX functions 
        raise NotImplementedError


# NOTE This is for oracle estimate - need to write out alternative *latent cost*
# TODO write get_cost_func_e_step that uses low rank latent - refactor get_cost_func_e_step to make more general
# - really should have latent/obs cost func be connected to model object on instantiation... so latent / obs models are abstract class with cost_func etc
def cost_func_full(ev, Sigma_ests, u):
    L = Sigma_ests.shape[-1]
    uuH = jnp.outer(u, u.conj())

    logL = 0
    Sigma_ests_projected = jnp.einsum('ij,knl->inl', uuH, Sigma_ests)
    trace_sum = jnp.trace(Sigma_ests_projected).sum()

    logL = -L*jnp.log(ev) - (1/ev)*trace_sum
    return -logL.real

def cost_func_full_eigvec(u, Sigma_ests, ev):
    return cost_func_full(ev, Sigma_ests, u)

def eigvec_optim(ev_est, Sigma_ests, u_init, step_size=1, num_steps=1, method='GD', ts=False):
    cost_func = partial(cost_func_full_eigvec, Sigma_ests=Sigma_ests, ev=ev_est)
    cost_grad = jax.grad(cost_func)
    nu = step_size
    renorm = True
    method = 'GD'

    u_ests = []
    costs = []
    u_est = u_init
    u_ests.append(u_est)
    c = cost_func(u_est)
    costs.append(c)
    for s in range(num_steps):
        g = cost_grad(u_est).conj()
        if method == 'Newton':
            raise NotImplementedError
            # H_real = cost_hess(u_est.real)
            # H_imag = cost_hess(u_est.imag)
            # H = (H_real - H_imag*1j).conj()

            # u_est = u_est - jnp.linalg.inv(H) @ g
        elif method == 'GD':
            u_est = u_est - nu*g
        else: 
            raise NotImplementedError

        if renorm is True:
            u_est = u_est/jnp.linalg.norm(u_est)
            # angle1 = jnp.angle(u_est)[0]
            # u_est = u_est * jnp.exp(-1j*angle1)

        u_ests.append(u_est)

        c = cost_func(u_est)
        costs.append(c)
    if ts is True:
        return u_ests, costs
    else:
        return u_est

